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
- **裁切重绘**：聚焦重绘旁新增独立按钮，使用「AI 裁切框」指定输入范围，与聚焦重绘互斥。框没有聚焦模式的 1 MP 限制，不添加最小上下文；请求使用主面板目标分辨率及普通重绘参数，「自动」按裁切框比例适配。实际请求尺寸仍遵循普通重绘的合法尺寸限制，不保证免费。
  - 先选区或绘制重绘涂抹，再调整裁切框；真实选区优先。发送框内原图和相应蒙版，返回结果缩回原位置，只贴回蒙版区域，不替换整框，也不改动画布尺寸。
  - 框外部分不发送、不贴回；框内没有重绘区域时阻止生成。使用此模式前需关闭已激活的图生图控制层。生成后框隐藏但继续有效，再次点击该按钮或删除框图层可退出，状态随作品保存。
  - 激活后目标分辨率改为检测裁切框：缩放框并松开鼠标后重新检测，单纯移动框保留手动目标值。「自动」「小」按框内区域比例计算；手动宽高只调整框内生成尺寸，不反向缩放框或改变画布。无关画布尺寸、选区或图生图来源变化不会抢走目标值；退出后恢复原来源。
- **NAI 专属控制层**(点"+"添加,按图层选择参考图,悬停可预览缩略图):
  - 图像来源左侧复选开关控制单图激活状态，默认开启；关闭后保留来源和参数，但不参与生成。未激活的图生图不接管主强度或目标尺寸，也不占用激活控制层数量限额。
  - 类别旁「+」追加同类参考图，沿用该行来源和参数，随后可分别选择素材与调整参数；同组只显示一次类别，子项图像列对齐，箭头展开/收起。原有添加控制层入口仍新建独立项，可分别管理角色 A、角色 B。
  - 分组后类别左侧出现整组开关，单图开关仍在图像旁；关组不改变单图原来的勾选，重开恢复。折叠只隐藏子项，不停用参考图。分组、展开与开关状态随作品保存；移除首图后下一张自动成为组首。
  - 分组仅组织 UI，不新增服务端角色绑定、不绕过模型能力或图片数量限制；图生图保持单输入，不支持追加同组底图。
  - **图生图**:选一个图层(取完整原图拉伸)或"整张画布"(截取画布窗口)作为底图;强度(默认 0.7)和噪声(默认 0)在展开面板里独立调节,与主滑块无关
  - **Vibe Transfer**:可挂多层,每层独立强度 / 信息提取度;V4 / V4.5 需经官方 `encode-vibe` 预编码(**每张图 2 Anlas**),编码结果按图片内容持久缓存,同一张图永远只付一次;V5 首发暂不支持,插件不会发起付费编码
  - **精准参考(Director Reference)**:角色 / 风格 / 角色&风格三种类型,强度 + 保真度可调;目前仅 V4.5 模型可用,V5 首发暂不支持
  - Vibe 与精准参考取图**严格用图层完整原图**(不裁剪到画布、不缩放)
- **多 Token 管理**:设置页可保存多个 NAI 账号 token 并切换,显示订阅等级与 Anlas 余额

## NAI 本机共享素材库（本地 UI/UX 扩展）

点击 NAI 控制项旁的 **相册图标**，或通过 **工具 → 脚本 → NAI 素材库…** 打开。
即使没有打开任何画布也可管理素材；图片独立保存在本机，不依赖当前画布或 KRA 文件，
关闭作品、放弃保存或重启 Krita 后仍保留，所有画布共享同一素材库。

- 可直接向素材库拖入图片，或通过「导入」批量添加；拖入画布的原生菜单也提供 **添加到 NAI 素材库**，不插入画布图层。
- 素材库采用可折叠的父子分类树，可新建子分类、重命名、拖动分类或素材归类，右键可移到顶层。删除分类仅将子分类和素材移到上一级，不删除图片。
- 点击分类整行即可展开或收起；「全部展开 / 全部收起」共用一个按钮，部分展开时先全部展开。关闭后记住各级状态，包括收起父级内的子分类，重启后仍保留。
- 素材树与来源下拉列表只显示名称，不显示尺寸，不添加 `[NAI]` 前缀；素材仍保留完整原图像素。
- 下拉列表不展示整个素材库，只显示当前控制项已经选中的那一张素材；通过相册选择或更换，切回画布、选区或普通图层后即移除素材选项。各控制项互不影响。
- 支持预览、重命名和删除；新素材不写入 KRA 附件、不参与画布合成，不影响画布边界和裁剪。
- 图生图、Vibe、精准参考均读取素材完整像素。图生图结果仍放置在当前画布范围，不以素材尺寸扩展画布。
- 支持 Qt 可读取的常见图片；KRA/ORA 输入读取其合成图，不导入图层树。导入后不依赖原文件路径。
- 数据位于插件用户数据目录的 `nai_references` 文件夹内，包含 JSON 索引、分类归属及 PNG 图片，与插件代码目录分开。
- 分类展开状态单独保存在同目录的 `ui_state.json`，不改动素材索引或 KRA。
- 分类索引使用版本 2，首次更新旧版索引前保存 `nai_references.v1-backup.json`；旧素材和原 UUID 自动兼容。
- **跨设备需迁移完整素材库目录**：单独复制 KRA 只保留来源 ID，不携带新素材。素材库本身没有云同步。
- 上版保存在 KRA 附件中的素材会在作品打开时自动复制到本机素材库，保留原始 ID 和原附件，不破坏旧作品。
- 删除是全局操作：会移除所有已打开作品中引用它的控制项，其他已保存作品会显示缺失提示；不删除画布图层，不进入画布撤销栈。
- 素材库损坏时不会覆盖原索引；文件缺失时会明确报错，不会悄悄改用当前画布。存储和解码仍有磁盘、内存开销。
- 已存在的超大普通图层不会自动转换或删除。仅有远程 URL、没有实际图像数据的浏览器拖入仍使用原生功能。

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

提示词框：Enter 生成，Shift+Enter 换行（支持小键盘回车）。补全列表打开时，Enter 先确认补全，Shift+Enter 仍直接换行。

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
