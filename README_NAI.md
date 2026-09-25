# Krita AI Diffusion — NovelAI 版

基于 [Acly/krita-ai-diffusion](https://github.com/Acly/krita-ai-diffusion)(v1.52.1)的分支,为 Krita 接入 **NovelAI 图像生成 API**(V4 / V4.5 / V5 系列模型),无需本地显卡和 ComfyUI。原插件的 ComfyUI / 云端功能保持不变,NovelAI 作为第四种后端并存。

参数与算法参考了 NovelAI 官网前端与 Aaalice NAI Launcher 的实现。

## 功能

- **文生图**:V4 / V4.5 / V5(curated / full)模型,支持风格预设、质量标签和负面预设等;V5 默认使用官方 23 步、CFG 7、Euler Ancestral、Karras 参数
- **局部重绘(Inpaint)**:画选区直接重绘,使用 NAI 官方 `-inpainting` 重绘模型
  - V5 Full 使用 `nai-diffusion-5-full-inpainting`;V5 Curated 的专用重绘模型尚未上线,按官网当前行为临时回退 V4.5 Curated Inpainting
  - 重绘幅度滑块真实有效(见下方"技术备注")
  - 蒙版严格按 8px latent 网格对齐,结果以透明补丁形式贴回画布——选区外像素零改动、无色偏、无黑边
- **整图重绘**:无选区时把强度滑块调到 100% 以下即可
- **NAI 专属控制层**(点"+"添加,按图层选择参考图,悬停可预览缩略图):
  - **图生图**:选一个图层(取完整原图拉伸)或"整张画布"(截取画布窗口)作为底图;强度(默认 0.7)和噪声(默认 0)在展开面板里独立调节,与主滑块无关
  - **Vibe Transfer**:可挂多层,每层独立强度 / 信息提取度;V4 / V4.5 需经官方 `encode-vibe` 预编码(**每张图 2 Anlas**),编码结果按图片内容持久缓存,同一张图永远只付一次;V5 首发暂不支持,插件不会发起付费编码
  - **精准参考(Director Reference)**:角色 / 风格 / 角色&风格三种类型,强度 + 保真度可调;目前仅 V4.5 模型可用,V5 首发暂不支持
  - Vibe 与精准参考取图**严格用图层完整原图**(不裁剪到画布、不缩放)
- **多 Token 管理**:设置页可保存多个 NAI 账号 token 并切换,显示订阅等级与 Anlas 余额

## 安装

推荐从 [Releases](https://github.com/floit04/krita-ai-diffusion-nai/releases) 下载最新的
`krita_ai_diffusion-*-nai*.zip`,在 Krita 里选 **工具 → 脚本 → 从文件导入 Python 插件**,
选中该 zip,然后重启 Krita。

也可以手动安装(从源码运行时不会收到自动更新提示):

1. 下载本仓库(Code → Download ZIP 或 `git clone`)
2. 把 `ai_diffusion` 文件夹和 `ai_diffusion.desktop` 复制到 Krita 的 pykrita 目录:
   - Windows:`%APPDATA%\krita\pykrita\`
   - Linux:`~/.local/share/krita/pykrita/`
3. 启动 Krita → 设置 → 配置 Krita → Python 插件管理器 → 勾选 **AI Image Diffusion** → 重启 Krita
4. 设置 → 面板 → 勾选 **AI Image Generation** 打开面板

## 更新

插件的自动更新指向**本仓库的 Releases**,而不是上游 Acly 的官方服务 —— 装了 NAI 版就只收
NAI 版的更新,与上游同步由本仓库手动合并后再发版。

启动时会检查一次(可在 齿轮 → **关于** 页关闭),有新版会提示;也可以在该页点
**Check for Updates** 手动检查,再点 **Download and Install** 就地升级,重启 Krita 生效。
下载完会用发布包附带的 `.sha256` 校验完整性。从源码目录运行时版本号显示为 `x.y.z-dev`,
自动更新不生效(以免覆盖你的工作副本)。

## 配置

Krita 的 AI 面板 → 右上角齿轮 → **连接** 页 → 选 **NovelAI** 标签,两种登录方式任选:

**方式一:账号密码登录(推荐)**
填邮箱和密码 → **登录**。插件在本地用 Argon2id 算出访问密钥,只把密钥发给 NovelAI 换取令牌 —— **密码本身不会离开你的电脑,也不会被保存**。密钥推导是纯 Python 实现的(Krita 自带的 Python 没有 argon2 库,也无法 pip 安装),约 1 秒。拿到的令牌有效期约 30 天,过期后重新登录即可。

**方式二:粘贴 Persistent Token**
在 [novelai.net](https://novelai.net) 登录后到 **Account → Get Persistent API Token** 复制 `pst-` 开头的令牌,粘贴后点连接。

登录成功后到 **NovelAI 风格** 页选择模型。默认推荐 `nai-diffusion-5-curated`;需要 V5 原生局部重绘时请选择 `nai-diffusion-5-full`。

> 令牌只保存在本机 Krita 配置目录(`%APPDATA%\krita\ai_diffusion\settings.json`),不会进入本仓库。**密码和邮箱一律不落盘**,登录框关掉就没了。

## 使用要点

| 想做什么 | 操作 |
|---|---|
| 文生图 | 无选区,强度 100%,点生成 |
| 整图重绘 | 无选区,强度调低(如 50%),点生成 |
| 局部重绘 | 画选区,主强度滑块=重绘幅度,点生成 |
| 图生图(垫图) | 控制层"+" → 图生图 → 选图层或"整张画布",展开面板调强度/噪声 |
| Vibe / 精准参考 | 控制层"+" → 对应类型 → 选图层,展开面板调参数 |

规则:有选区时以重绘优先(图生图层被忽略);Vibe 与精准参考同时存在时按官方行为保留精准参考;重绘时 Vibe 自动丢弃(NAI 服务端限制)。

## 手动图层颜色匹配（Windows，nai14）

强度右侧、画笔左侧的调色盘按钮处理当前普通绘画图层，参考下方可见图层同坐标合成画面。再次点击恢复原色，重新开启会重新取参考。绿色对号表示当前会话已匹配。不自动处理生成结果，不改变原有 Color Match。

**不支持原生 Ctrl+Z，请先复制图层。** 恢复状态仅当前会话有效，后续像素或位置变化会拒绝覆盖。原始像素备份留在 `%LOCALAPPDATA%\KritaColorMatch\layer-snapshots`，不会自动删除。支持 RGB/Alpha U8、图层与文档相同 profile；锁定、隐藏、动画和变形蒙版等情况不处理。非 Normal 混合模式不保证视觉匹配效果。

需要独立 Python 3.10 或更高版本，勿更改 Krita 内置 Python。在 PowerShell 中运行一次（以本机已有 3.10 为例）：

```powershell
py -3.10 -m venv "$env:LOCALAPPDATA\KritaColorMatch\venv"
& "$env:LOCALAPPDATA\KritaColorMatch\venv\Scripts\python.exe" -m pip install color-matcher==0.6.0 numpy pillow
$cfg = @{python="$env:LOCALAPPDATA\KritaColorMatch\venv\Scripts\python.exe"; method='hm-mvgd-hm'; backend_ranges=@()} | ConvertTo-Json
[IO.File]::WriteAllText("$env:LOCALAPPDATA\KritaColorMatch\runtime.json", $cfg, [Text.UTF8Encoding]::new($false))
```

计算默认使用 CPU，无需 ComfyUI、生图 API 或 GPU。算法沿用交接中的 `ColorMatcher().transfer(method="hm-mvgd-hm")` 调用；可选 GPU 仅加速直方图阶段，失败回退 CPU。不要复制其他机器的 Python 绝对路径、虚拟环境或 GPU 阈值。发布包不含计算依赖、配置或图层备份。

## 技术备注(与其它第三方实现的差异)

- **重绘强度的真实字段**:NAI 服务端只认嵌套对象 `parameters.img2img = {"strength": …, "color_correct": true}`(逆向官网前端 bundle 所得);常见的扁平字段 `inpaintImg2ImgStrength` 会被服务端忽略。目前所见的社区实现(各类启动器 / ComfyUI 节点 / API 封装库)均只发送扁平字段,其重绘强度实际不生效。
- **账号端点**:`/user/subscription` 等 `/user/*` 端点必须请求 `image.novelai.net`;`api.novelai.net` 对第三方工具返回 400。
- **infill 不能走流式接口**:`generate-image-stream` 会忽略蒙版把请求当整图 img2img,重绘必须用非流式 `generate-image`。
- Vibe 编码(`/ai/encode-vibe`)按 (图片哈希, 模型, 信息提取度) 缓存于 `%APPDATA%\krita\ai_diffusion\nai_vibe_cache.json`。

## 致谢

- [Acly/krita-ai-diffusion](https://github.com/Acly/krita-ai-diffusion) — 原插件
- Aaalice NAI Launcher — NAI 请求参数与蒙版处理的参考实现
- [NovelAI](https://novelai.net) — 图像生成服务

本分支与 NovelAI 官方无关;使用请遵守 NovelAI 服务条款。
