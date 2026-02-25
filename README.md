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

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd4c3f0f-4ac2-4d59-910d-4dec2d786b4f" width="100%" /> 
</p>

> - *OpenUtau 版本来自 [keirokeer/OpenUtau-DiffSinger-Lunai](https://github.com/keirokeer/OpenUtau-DiffSinger-Lunai)*
> - *歌手模型来自 [yousa-ling-official-production/yousa-ling-diffsinger-v1](https://github.com/yousa-ling-official-production/yousa-ling-diffsinger-v1)*

## ✅ 支持平台

* Windows / Linux
* OpenUtau Beta（支持 DiffSinger）
* Python 3.10 \*

若您的系统中安装了 NVIDIA 显卡驱动，且驱动支持 [CUDA 11.x](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)（即：驱动版本 >= 450），本应用会自动启用 GPU 加速以提升处理速度。


<details>
  <summary>点击展开更多平台说明</summary>

* \* 本项目使用 [CREPE](https://github.com/marl/crepe) \*\* 作为音高提取器，依赖于 TensorFlow 框架。在 Windows 平台下，TensorFlow 2.10 是最后一个支持 GPU 加速的版本，Python 3.10 是它的 `.whl` 文件支持的最高 Python 版本。
* \*\* 未来可能切换至基于 PyTorch 的 [PESTO](https://github.com/SonyCSLParis/pesto) 替代 CREPE，以提高兼容性。

</details>

## 📌 使用场景

**典型需求**：在使用 DiffSinger 虚拟歌手翻唱时，已经完成了填好词的无参 OpenUtau 工程，但尚未添加表情参数。本应用可以从参考人声音频中提取表情参数，并导入至 OpenUtau 工程中。

**所需输入：**

* 歌姬音声：由 OpenUtau 输出的无表情虚拟歌声音频（WAV 格式）。建议节奏 (`Tempo`) 和分段尽量与参考人声一致。
* 参考人声：原始人声录音（WAV 格式），可使用 [UVR](https://github.com/Anjok07/ultimatevocalremovergui) 等工具去除伴奏与混响。
* 输入工程：原始 OpenUtau 工程文件（USTX 格式）。
* 输出工程路径：处理完成后新工程文件的保存位置。

**输出结果：**

* 一个携带表情参数的新 USTX 文件，原始工程不会被修改。

## ✨ 功能特性

* [x] Windows 支持
* [x] Linux 支持
* [x] NVIDIA GPU 加速
* [x] 参数配置导入 / 导出
* [x] `Pitch Deviation` 参数生成
* [x] `Dynamics` 参数生成
* [x] `Tension` 参数生成

## ⚠️ 已知问题

1. 当前版本尚不支持单一轨道中的 `Tempo` 变化，建议工程全程使用统一节奏。该限制将在未来版本中解决。

## 🚀 直接安装

您可以直接在 [Releases](https://github.com/NewComer00/expressive/releases) 页面下载预编译的可执行文件:

- `Expressive-GUI-<version>-Windows-x64-CPU.exe`: 适用于 x64 架构 Windows 的图形用户界面安装包（仅 CPU，无 CUDA 依赖）
- `Expressive-GUI-<version>-Windows-x64-GPU.exe`: 适用于 x64 架构 Windows 的图形用户界面安装包（含 NVIDIA GPU 加速，需驱动版本 >= 450，支持 CUDA 11.x）

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

## 🔬 算法流程
```mermaid
graph TB;
  ustx_in[/"OpenUtau Project (USTX)"/]
  refwav[/"Reference WAV"/]
  utauwav[/"OpenUtau WAV"/]
  refwav-->feat_pitd
  ustx_in-.->|Export|utauwav
  utauwav-->feat_pitd
  ustx_in-->|Tempo|time_pitd

  subgraph PitdLoader
    direction TB
    feat_pitd["Features Extraction<br>Pitch & MFCC"]

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
