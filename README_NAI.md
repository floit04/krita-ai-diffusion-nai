# Krita AI Diffusion — NovelAI 版

基于 [Acly/krita-ai-diffusion](https://github.com/Acly/krita-ai-diffusion)(v1.52.1)的分支,为 Krita 接入 **NovelAI 图像生成 API**(V4 / V4.5 系列模型),无需本地显卡和 ComfyUI。原插件的 ComfyUI / 云端功能保持不变,NovelAI 作为第四种后端并存。

参数与算法参考了 NovelAI 官网前端与 Aaalice NAI Launcher 的实现。

## 功能

- **文生图**:V4 / V4.5(curated / full)模型,支持风格预设、质量标签、负面预设、Variety+、SMEA 等
- **局部重绘(Inpaint)**:画选区直接重绘,使用 NAI 官方 `-inpainting` 重绘模型
  - 重绘幅度滑块真实有效(见下方"技术备注")
  - 蒙版严格按 8px latent 网格对齐,结果以透明补丁形式贴回画布——选区外像素零改动、无色偏、无黑边
- **整图重绘**:无选区时把强度滑块调到 100% 以下即可
- **NAI 专属控制层**(点"+"添加,按图层选择参考图,悬停可预览缩略图):
  - **图生图**:选一个图层(取完整原图拉伸)或"整张画布"(截取画布窗口)作为底图;强度(默认 0.7)和噪声(默认 0)在展开面板里独立调节,与主滑块无关
  - **Vibe Transfer**:可挂多层,每层独立强度 / 信息提取度;V4+ 需经官方 `encode-vibe` 预编码(**每张图 2 Anlas**),编码结果按图片内容持久缓存,同一张图永远只付一次
  - **精准参考(Director Reference)**:角色 / 风格 / 角色&风格三种类型,强度 + 保真度可调;仅 V4.5 模型可用
  - Vibe 与精准参考取图**严格用图层完整原图**(不裁剪到画布、不缩放)
- **多 Token 管理**:设置页可保存多个 NAI 账号 token 并切换,显示订阅等级与 Anlas 余额

## 安装

1. 下载本仓库(Code → Download ZIP 或 `git clone`)
2. 把 `ai_diffusion` 文件夹和 `ai_diffusion.desktop` 复制到 Krita 的 pykrita 目录:
   - Windows:`%APPDATA%\krita\pykrita\`
   - Linux:`~/.local/share/krita/pykrita/`
3. 启动 Krita → 设置 → 配置 Krita → Python 插件管理器 → 勾选 **AI Image Diffusion** → 重启 Krita
4. 设置 → 面板 → 勾选 **AI Image Generation** 打开面板

## 配置

1. 在 [novelai.net](https://novelai.net) 登录后到 **Account → Get Persistent API Token** 复制 `pst-` 开头的令牌
2. Krita 的 AI 面板 → 右上角齿轮 → **连接** 页 → 选 **NovelAI** 标签 → 粘贴令牌 → 连接
3. 在 **NovelAI 风格** 页选择模型(推荐 `nai-diffusion-4-5-full`)并调整采样参数

> 令牌只保存在本机 Krita 配置目录(`%APPDATA%\krita\ai_diffusion\settings.json`),不会进入本仓库。

## 使用要点

| 想做什么 | 操作 |
|---|---|
| 文生图 | 无选区,强度 100%,点生成 |
| 整图重绘 | 无选区,强度调低(如 50%),点生成 |
| 局部重绘 | 画选区,主强度滑块=重绘幅度,点生成 |
| 图生图(垫图) | 控制层"+" → 图生图 → 选图层或"整张画布",展开面板调强度/噪声 |
| Vibe / 精准参考 | 控制层"+" → 对应类型 → 选图层,展开面板调参数 |

规则:有选区时以重绘优先(图生图层被忽略);Vibe 与精准参考同时存在时按官方行为保留精准参考;重绘时 Vibe 自动丢弃(NAI 服务端限制)。

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
