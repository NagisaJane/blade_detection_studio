# -*- coding: utf-8 -*-
"""
AUD-YOLO 改进模块（结合论文 An improved method of AUD-YOLO for surface damage detection of wind turbine blades）
- ADown: 替代部分 backbone 下采样，保留细节、减参
- C2f_UniRepLK: 颈部大核卷积，与 C2f 融合
- DySample: 动态上采样替代 nearest，保留语义
在 ultralytics YOLOv8 上注册并可用 yaml 构建模型。
"""

from ultralytics.nn.modules.conv import Conv
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import os

# ===================== ADown（YOLOv9 风格自适应下采样） =====================
class ADown(nn.Module):
    """自适应下采样：avg_pool → 通道切分 → 分支1 conv3x3 s2，分支2 maxpool+conv1x1 → concat。"""

    def __init__(self, c1: int, c2: int, *args, **kwargs):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = F.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


# ===================== DySample（轻量动态上采样，替代 nn.Upsample） =====================
class DySample(nn.Module):
    """scale=2 动态上采样。ultralytics 对自定义层只传 yaml 的 args，即 m(2)，故 c1=2；首次 forward 时用实际输入通道延迟构建 offset。"""

    def __init__(self, c1: int, c2=None, scale: int = 2, style: str = "lp", groups: int = 4, *args, **kwargs):
        super().__init__()
        # ultralytics 传 m(2) 时 c1=2，当作 scale，延迟到 forward 再建 offset
        if c1 in (1, 2, 4) and (c2 is None or int(c2) in (1, 2, 4)):
            self._scale = int(c1) if c1 in (2, 4) else 2
            self._defer_build = True
            self.offset = None
            self.scale = self._scale
            self.style = style
            self.groups = max(1, min(4, groups))
            self.register_buffer("init_pos", self._init_pos_buffer(self._scale, self.groups))
            return
        if c2 is not None and int(c2) in (1, 2, 4):
            scale = int(c2)
            c2 = c1
        self._defer_build = False
        self.c2 = c1 if c2 is None else c2
        self.scale = scale
        self.style = style
        self.groups = min(max(1, groups), c1)
        if self.style == "pl":
            assert c1 >= scale ** 2 and c1 % scale ** 2 == 0
        assert c1 >= self.groups and c1 % self.groups == 0
        if style == "pl":
            in_ch = c1 // scale ** 2
            out_ch = 2 * self.groups
        else:
            in_ch = c1
            out_ch = 2 * self.groups * scale ** 2
        self.offset = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0)
        self.register_buffer("init_pos", self._init_pos())

    @staticmethod
    def _init_pos_buffer(scale: int, groups: int) -> torch.Tensor:
        h = torch.arange((-scale + 1) / 2, (scale - 1) / 2 + 1e-5) / scale
        grid = torch.stack(torch.meshgrid([h, h], indexing="ij")).transpose(1, 2)
        return grid.repeat(1, groups, 1).reshape(1, -1, 1, 1)

    def _init_pos(self):
        return self._init_pos_buffer(self.scale, self.groups)

    def _sample(self, x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(W, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.view(B, -1, H, W)
        coords = F.pixel_shuffle(coords, self.scale)
        coords = coords.view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        out = F.grid_sample(
            x.reshape(B * self.groups, -1, H, W), coords, mode="bilinear", align_corners=False, padding_mode="border"
        )
        return out.view(B, -1, self.scale * H, self.scale * W)

    def _build_offset(self, x: torch.Tensor):
        """ultralytics 只传 m(2) 时在首次 forward 用实际通道数构建 offset。"""
        self._defer_build = False
        c1 = x.shape[1]
        scale, style = self.scale, self.style
        groups = min(max(1, self.groups), c1)
        if style == "pl":
            in_ch = c1 // (scale ** 2)
            out_ch = 2 * groups
        else:
            in_ch = c1
            out_ch = 2 * groups * scale ** 2
        self.offset = nn.Conv2d(in_ch, out_ch, 1).to(x.device)
        nn.init.normal_(self.offset.weight, std=0.001)
        if self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0)

    def forward_lp(self, x: torch.Tensor) -> torch.Tensor:
        if self._defer_build:
            self._build_offset(x)
        offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)

    def forward_pl(self, x: torch.Tensor) -> torch.Tensor:
        if self._defer_build:
            self._build_offset(x)
        x_ = F.pixel_shuffle(x, self.scale)
        offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self._sample(x, offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)


# ===================== C2f_UniRepLK（颈部大核 C2f） =====================
class SEBlock(nn.Module):
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(c // r, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(c, mid, 1), nn.SiLU(),
            nn.Conv2d(mid, c, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class RepLKBlock(nn.Module):
    """大核块：5x5 DW + PW + SE，用于 C2f 内替代 Bottleneck。"""

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        c_ = max(1, int(c2 * 0.5))  # groups=c_ 要求 >0，width scale 后 c2 可能很小
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_, c_, 5, 1, 2, groups=c_),
            nn.Conv2d(c_, c2, 1, 1),
        )
        self.se = SEBlock(c2, 16)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.se(self.cv2(self.cv1(x)))
        return (x + out) if self.add else out


class C2f_UniRepLK(nn.Module):
    """C2f 内用 RepLKBlock，用于 neck 大感受野。若实际输入通道与构建时 c1 不一致，用 1x1 投影兜底。"""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5, *args, **kwargs):
        super().__init__()
        n, c1, c2 = int(n), int(c1), int(c2)
        self._c1 = c1
        self.c = max(1, int(c2 * e))
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(RepLKBlock(self.c, self.c, shortcut) for _ in range(n))
        self._proj = None  # 懒创建：当输入通道 != c1 时用 1x1 投影到 c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 实际输入通道与 cv1 不一致时（如服务器 ch 列表错误），用 1x1 投影到 cv1 的 in_channels
        need_c1 = self.cv1.conv.in_channels
        if x.shape[1] != need_c1:
            if getattr(self, "_proj", None) is None:
                self._proj = Conv(x.shape[1], need_c1, 1, 1).to(device=x.device)
            x = self._proj(x)
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# ===================== Concat 兜底：空间尺寸不一致时 resize 再 concat =====================
class Concat(nn.Module):
    """与 ultralytics Concat 接口一致；当多路特征图空间尺寸不一致时，先 interpolate 到第一路尺寸再 concat，避免 64/16 等报错。"""

    def __init__(self, dimension=1, *args, **kwargs):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) <= 1:
            return torch.cat(x, self.d) if isinstance(x, (list, tuple)) else x
        target_h, target_w = x[0].shape[2], x[0].shape[3]
        out = []
        for t in x:
            if t.shape[2] != target_h or t.shape[3] != target_w:
                t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
            out.append(t)
        return torch.cat(out, self.d)


# ===================== 注册到 ultralytics parse_model =====================
def register_aud_modules():
    import sys
    import ultralytics.nn.tasks as tasks_module

    # AUD 模块（覆盖或注入）
    tasks_module.ADown = ADown  # 用我们的 ADown 替代官方的，签名一致 (c1, c2)
    tasks_module.DySample = DySample
    tasks_module.C2f_UniRepLK = C2f_UniRepLK
    tasks_module.Concat = Concat

    _demo_dir = os.path.dirname(os.path.abspath(__file__))
    if _demo_dir not in sys.path:
        sys.path.insert(0, _demo_dir)
    try:
        from train_rig_yolov8n import RFAConv as _RFAConv, C2f_IDC as _C2f_IDC, GSA as _GSA, IDCBlock as _IDCBlock
        tasks_module.RFAConv = _RFAConv
        tasks_module.C2f_IDC = _C2f_IDC
        tasks_module.GSA = _GSA
        tasks_module.IDCBlock = _IDCBlock
    except ImportError:
        pass

    try:
        try:
            src = inspect.getsource(tasks_module.parse_model)
        except OSError:
            # 服务器/打包环境可能拿不到 exec 后函数的源码，改为从 tasks.py 文件读取 parse_model
            path = inspect.getfile(tasks_module)
            with open(path, "r", encoding="utf-8") as f:
                full = f.read()
            beg = full.find("def parse_model(")
            if beg == -1:
                raise RuntimeError("AUD 注入失败：在 tasks 中未找到 parse_model")
            end = full.find("\ndef ", beg + 1)
            if end == -1:
                end = len(full)
            src = full[beg:end]

        if "C2f_UniRepLK" in src and "elif m is DySample:" in src:
            return  # 已注入过

        # 1) C2f_UniRepLK 加入 base_modules → 得到 c1=ch[f], c2=args[0], args=[c1,c2,*args[1:]]
        src = src.replace(
            "            C2f,\n            C3k2,\n            RepNCSPELAN4,",
            "            C2f,\n            C2f_UniRepLK,\n            C3k2,\n            RepNCSPELAN4,",
        )
        # 2) C2f_UniRepLK 加入 repeat_modules → args.insert(2, n)
        src = src.replace(
            "            C2f,\n            C3k2,\n            C2fAttn,",
            "            C2f,\n            C2f_UniRepLK,\n            C3k2,\n            C2fAttn,",
        )
        # 3) DySample 需要 args=[ch[f], *args] 且 c2=ch[f]（输出通道=输入通道），用 elif 插在 else 前
        _old_else = "        else:\n            c2 = ch[f]"
        _new_else = (
            "        elif m is DySample:\n"
            "            args = [ch[f], *args]\n"
            "            c2 = ch[f]\n"
            "        else:\n"
            "            c2 = ch[f]"
        )
        if _old_else in src and "elif m is DySample:" not in src:
            src = src.replace(_old_else, _new_else)

        if "elif m is DySample:" not in src:
            raise RuntimeError("AUD 注入失败：未找到 else c2=ch[f] 位置，请检查 ultralytics 版本")
        ns = dict(tasks_module.__dict__)
        exec(src, ns)
        tasks_module.parse_model = ns["parse_model"]
    except Exception as e:
        raise RuntimeError("AUD 注入 parse_model 失败: " + str(e)) from e


if __name__ == "__main__":
    # 可选：缓解验证阶段绘图时「Glyph xxx missing from font DejaVu Sans」中文乱码/警告（需系统有中文字体）
    try:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    register_aud_modules()
    from ultralytics import YOLO

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(demo_dir, "aud_yolov8n.yaml")
    data_yaml = os.path.join(demo_dir, "dataset.yaml")

    # 腾讯云/显存不足时可用环境变量减小占用：AUD_BATCH=4 AUD_IMGSZ=640
    batch = int(os.environ.get("AUD_BATCH", "8"))
    imgsz = int(os.environ.get("AUD_IMGSZ", "640"))

    if not os.path.isfile(yaml_path):
        print("请先在同一目录下创建 aud_yolov8n.yaml（使用 ADown / DySample / C2f_UniRepLK）")
    else:
        print("加载 AUD-YOLOv8n（ADown + C2f_UniRepLK + DySample）...")
        model = YOLO(yaml_path)
        print("启动训练（batch=%d, imgsz=%d）..." % (batch, imgsz))
        model.train(
            data=data_yaml,
            epochs=450,
            batch=batch,
            lr0=0.01,
            lrf=0.01,
            optimizer="SGD",
            imgsz=imgsz,
            weight_decay=0.0005,
            momentum=0.937,
            patience=120,
            save_period=50,
            val=True,
            cache=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        val_results = model.val()
        # box.p / box.r / box.map50 可能是 per-class 的 ndarray，需转成标量再格式化
        def _to_float(x):
            if x is None:
                return 0.0
            if hasattr(x, "mean"):
                return float(x.mean())
            return float(x)
        p = _to_float(getattr(val_results.box, "mp", None) or val_results.box.p)
        r = _to_float(getattr(val_results.box, "mr", None) or val_results.box.r)
        m50 = _to_float(getattr(val_results.box, "map50", None))
        print("验证：P: {:.4f}  R: {:.4f}  mAP50: {:.4f}".format(p, r, m50))
