<p align="center">
  <img src="assets/splash/normal.png" width="50%" /> 
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/语言-简体中文-red.svg"></a>
  <a href="README.en.md"><img src="https://img.shields.io/badge/lang-English-blue.svg"></a>
</p>

# Expressive

**Expressive** 是一个为 [OpenUtau](https://github.com/stakira/OpenUtau) 开发的 [DiffSinger](https://github.com/openvpi/diffsinger) 表情参数导入工具，旨在从真实人声中提取表情参数，并导入至工程的相应轨道。

当前版本支持以下表情参数的导入：

* `Dynamics (curve)`
* `Pitch Deviation (curve)`
* `Tension (curve)`

<p align="middle">
  <img src="https://github.com/user-attachments/assets/268b44d4-528d-481e-acfb-3f7da7261c80" width="100%" /> 
</p>

> - *OpenUtau 版本来自 [keirokeer/OpenUtau-DiffSinger-Lunai](https://github.com/keirokeer/OpenUtau-DiffSinger-Lunai)*
> - *歌手模型来自 [yousa-ling-official-production/yousa-ling-diffsinger-v1](https://github.com/yousa-ling-official-production/yousa-ling-diffsinger-v1)*

> [!TIP]
> <details>
>   <summary><b>👉 点击展开完整有声演示视频 👈</b></summary>
>
>   <p align="center"><video src="https://github.com/user-attachments/assets/4b5b7c15-947a-4f54-b80e-a14a9eefc86b"></video></p>
> 
> </details>

## ✅ 支持平台

* Windows / Linux
* OpenUtau Beta（或支持 DiffSinger 的其他版本）
* Python 3.10 \*

本应用默认选择 [swift-f0](https://github.com/lars76/swift-f0)（基于 ONNX Runtime）作为音高提取后端，仅需 CPU 即可运行，可满足基础使用场景。

也提供了经典的 [CREPE](https://github.com/marl/crepe)（依赖 TensorFlow）音高提取后端，适合更高要求的使用场景。若您的电脑配有 NVIDIA 显卡且支持 [CUDA 11.x](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)（即显卡驱动版本 >= 450），使用 CREPE 后端时会自动启用 GPU 加速。

> \* 在 Windows 平台下，TensorFlow 2.10 是最后一个支持 GPU 加速的版本，Python 3.10 是它的 `.whl` 文件支持的最高 Python 版本。

## 📌 使用场景

### 需求

在使用 DiffSinger 虚拟歌手翻唱时，已经完成了填好词的无参 OpenUtau 工程，但尚未添加表情参数。本应用可以从参考人声音频中提取表情参数，并导入至 OpenUtau 工程中。

### 输入

> [!TIP]
> 从 `v0.6.0` 开始，本应用支持带有**多分段**与**多曲速**的 OpenUtau 人声音轨。

> [!TIP]
> 从 `v0.5.0` 开始，用户可以分别在**歌姬音声**与**参考人声**的完整音频中划定选区，选区内的音频段落将作为最终输入。

* **歌姬音声**：由 OpenUtau 输出的无表情虚拟歌声音频（WAV 格式）。建议分段与曲速尽量与**参考人声**相近，若相差过大可能影响对齐效果。
* **参考人声**：原始人声录音（WAV 格式），可使用 [UVR](https://github.com/Anjok07/ultimatevocalremovergui) 、[MSST](https://github.com/SUC-DriverOld/MSST-WebUI) 等工具去除伴奏、和声与混响。
* **输入工程**：原始 OpenUtau 工程文件（USTX 格式）。
* **输出路径**：处理完成后新工程文件的保存位置。
* **音轨编号**：OpenUtau 工程中**歌姬音声**所在的音轨编号（从 1 开始）。表情参数会被导入到该音轨中。

### 输出

一个携带表情参数的新 USTX 文件。原始工程不会被修改。

## ✨ 功能特性

* [x] Windows 支持
* [x] Linux 支持
* [x] NVIDIA GPU 加速
* [x] 参数配置导入 / 导出
* [x] `Pitch Deviation` 参数生成
* [x] `Dynamics` 参数生成
* [x] `Tension` 参数生成

## 🚀 直接安装

您可以直接在 [Releases](https://github.com/NewComer00/expressive/releases) 页面下载预编译的可执行文件:

### `Expressive-GUI-<version>-Windows-x64-CPU.exe`
适用于 x64 架构 Windows 的图形用户界面安装包。

仅可使用 CPU，无 CUDA 运行时库。安装体积小，但选择 CREPE 后端提取音高时速度较慢。

### `Expressive-GUI-<version>-Windows-x64-GPU.exe`
带 GPU 支持的适用于 x64 架构 Windows 的图形用户界面安装包。

含 CUDA 运行时库。在配备 NVIDIA 显卡（驱动版本 >= 450）的电脑上使用时，会大幅提高 CREPE 后端的推理速度。

## 👨‍💻 源码安装

### 克隆仓库

> [!IMPORTANT]
> 本项目使用 [Git LFS](https://git-lfs.com/) 存储 `examples/` 下的示例音频等大文件。克隆仓库前，请确保本地已正确安装 Git LFS。

```bash
git clone https://github.com/NewComer00/expressive.git --depth 1
cd expressive
```

### 安装应用

请在虚拟环境中安装本软件包及其依赖：

```bash
pip install -e ".[gpu,gui]"
```

> [!TIP]
> - `-e` 参数以可编辑模式安装，便于二次开发
> - 本软件包支持的可选依赖项包括：
>   - `gpu`：启用 GPU 加速相关依赖（如 CUDA 运行时库）
>   - `gui`：启用图形用户界面相关依赖（如 NiceGUI）
>   - `dev`：开发环境依赖（如测试框架 pytest）
>   - `all`：安装上述所有依赖项

安装完成后，您将能够使用 `expressive` 和 `expressive-gui` 两个入口点来运行命令行界面和图形用户界面。

## 📖 使用方式

### 命令行界面（CLI）

显示帮助信息

```bash
expressive --help
```

在 Windows PowerShell 中执行示例命令

```powershell
expressive `
  --utau_wav "examples/明天会更好/utau.wav" `
  --ref_wav "examples/明天会更好/reference.wav" `
  --ustx_input "examples/明天会更好/project.ustx" `
  --ustx_output "examples/明天会更好/output.ustx" `
  --track_number 1 `
  --expression dyn `
  --expression pitd `
  --pitd.semitone_shift 0 `
  --expression tenc
```

在 Linux Shell 中执行示例命令

```bash
expressive \
  --utau_wav "examples/明天会更好/utau.wav" \
  --ref_wav "examples/明天会更好/reference.wav" \
  --ustx_input "examples/明天会更好/project.ustx" \
  --ustx_output "examples/明天会更好/output.ustx" \
  --track_number 1 \
  --expression dyn \
  --expression pitd \
  --pitd.semitone_shift 0 \
  --expression tenc
```

输出工程文件将保存在 `examples/明天会更好/output.ustx`。

### 图形用户界面（GUI）

启动中文界面

```bash
expressive-gui --lang zh_CN
```

> [!IMPORTANT]
> 由于框架限制，通过 `expressive-gui` 命令启动的图形界面目前**不支持文件拖放**功能。若需使用拖放功能，请[直接安装](#-直接安装)本应用，或以脚本方式运行 `expressive_gui.py`：
> 
> ```bash
> python expressive_gui.py --lang zh_CN
> ```

## 📂 示例工程

项目的 [`examples/` 目录](examples/)下存放有多个示例。您可以在图形用户界面中导入相应示例的 `expressive_config.json` 配置文件，将预设的参数一键填写到应用中。

若您是从安装包获取的本应用，安装完毕后示例目录的快捷方式 `Expressive-examples` 将出现在您的桌面，您也可以直接导入其中的配置文件。

## 🔬 算法流程
```mermaid
graph TB;
  ustx_in[/"OpenUtau Project (USTX)"/]
  refwav[/"Reference WAV"/]
  utauwav[/"OpenUtau WAV"/]
  refwav-->feat_pitd
  ustx_in-.->|Export|utauwav
  utauwav-->feat_pitd

  ustx_editor["USTX Editor"]
  ustx_in-->ustx_editor
  ustx_editor-->|UProject & Time Axis|PitdLoader

  subgraph PitdLoader
    direction TB
    feat_pitd["Features Extraction<br>Pitch & MFCC & RMS"]

    time_pitd["Time Alignment<br>FastDTW"]
    feat_pitd-->time_pitd

    pitch_algn["Pitch Alignment"]
    time_pitd-->pitch_algn

    get_pitd["Get Pitch Deviation"]
    pitch_algn-->get_pitd
  end

  utsx_out[/"OpenUtau Project Output"/]
  get_pitd-->utsx_out

  subgraph DynLoader
    direction TB
    feat_dyn["Features Extraction<br>RMS"]

    time_dyn["Time Alignment<br>FastDTW"]
    feat_dyn-->time_dyn

    get_dyn["Get Dynamics"]
    time_dyn-->get_dyn
  end

  subgraph TencLoader
    direction TB
    feat_tenc["Features Extraction<br>RMS"]

    time_tenc["Time Alignment<br>FastDTW"]
    feat_tenc-->time_tenc

    get_tenc["Get Tension"]
    time_tenc-->get_tenc
  end
```

## ⚠️ 常见问题

### 安装后首次运行图形界面，文件拖拽功能无法正常使用

#### 问题现象
在 Windows 10 / 11 平台下，通过安装包**首次**安装本应用后（先前安装过再卸载不算），应用的文件拖拽功能无法正常使用。

#### 可能原因
[NiceGUI](https://nicegui.io/) 框架对原生应用的文件拖拽功能支持尚不完善。目前本应用的文件拖拽功能是基于底层库 [pywebview](https://pywebview.flowrl.com/) 实现的。

#### 解决方案
重新打开应用后应当可以恢复正常，且该系统今后不会再出现此问题。

#### 未来计划
NiceGUI 框架已经开始着手改进文件拖拽支持，应该在未来的版本中能够解决此问题。

### PITD 表情曲线整体变化过于平缓

#### 问题现象
提取出的 PITD 表情曲线过于平缓，整体上几乎没有大的起伏，参考人声中的音高变化并没有反映到表情曲线上。

#### 可能原因
PITD 表情提取器中，两个置信度阈值设置**过高**，许多音高变化没有被采信。

#### 解决方案
尝试降低两个置信度阈值。一般来说，**歌姬音声**比较纯净，可以先调整**参考人声**的置信度阈值。

#### 未来计划
引入更好的 PITD 后端（如 [RMVPE](https://github.com/Dream-High/RMVPE)）。添加中间结果的可视化功能。

### PITD 表情曲线在某些位置变化过快，出现跳跃或毛刺

#### 问题现象
PITD 表情曲线在某些位置变化过快，出现非常大的跳跃或毛刺，明显不符合人声的变化规律。

#### 可能原因
PITD 表情提取器中，两个置信度阈值设置**过低**，错误的识别结果被采信。

#### 解决方案
尝试增加两个置信度阈值。一般来说，**歌姬音声**比较纯净，可以先调整**参考人声**的置信度阈值。

#### 未来计划
引入更好的 PITD 后端（如 [RMVPE](https://github.com/Dream-High/RMVPE)）。添加中间结果的可视化功能。
