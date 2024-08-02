[![Official](https://img.shields.io/badge/Official%20-Auromix-blue?style=flat&logo=world&logoColor=white)](https://github.com/Auromix) &nbsp;
[![Ubuntu VERSION](https://img.shields.io/badge/Ubuntu-22.04-green)](https://ubuntu.com/) &nbsp; [![LICENSE](https://img.shields.io/badge/license-Apache--2.0-informational)](https://github.com/Auromix/auro_puppeteer/blob/main/LICENSE) &nbsp;
[![GitHub Repo stars](https://img.shields.io/github/stars/Auromix/auro_puppeteer?style=social)](https://github.com/Auromix/auro_puppeteer/stargazers) &nbsp;
[![Twitter Follow](https://img.shields.io/twitter/follow/Hermanye233?style=social)](https://twitter.com/Hermanye233) &nbsp;

# 🤹‍♂️ Auro Puppeteer

An application leveraging Apple Vsion Pro and RGB camera for showcasing and teleoperation of robots.

![rgb_teleoperation](images/rgb_camera_dual_hand_teleoperation.gif)

# ⚙️ Install

`Auro Puppeteer` could be installed with following commands.

```bash
# Clone repo
git clone git@github.com:Auromix/auro_puppeteer.git
# Update submodules
cd auro_puppeteer
git submodule update --init --recursive
```

```bash
# Download latest assets to top directory of project
xdg-open https://drive.google.com/drive/folders/1bhe3dcdHjzunkdBjNsx2vrv3O-L8qrY_?usp=drive_link
```

```bash
# Create conda environment
conda create -n auro_puppeteer python=3.10 -y
conda activate auro_puppeteer
```

```bash
# Install dex-retargeting
cd auro_puppeteer/auro_puppeteer/libs/dex-retargeting/
pip  install -e ".[example]"
```

```bash
# Check GLIBC>2.34
ldd --version
# Install isaac sim in conda
pip install isaacsim==4.1.0.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-extscache-physics==4.1.0.0 isaacsim-extscache-kit==4.1.0.0 isaacsim-extscache-kit-sdk==4.1.0.0 --extra-index-url https://pypi.nvidia.com
```

```bash
# Install auro_puppeteer
cd auro_puppeteer
pip install -e .
```

# 🔥 Quickstart

You can find detailed examples in `auro_puppeteer/examples`.

## RGB Camera teleoperation

For RGB camera dual hand teleoperation, run:

```bash
python rgb_camera_dual_hand_teleoperation.py
```

# 🙋 Troubleshooting

For any issues, questions, or contributions related to this package,
please contact maintainers of this repository.

# 🧑‍💻 Future Development Plans

- Native Support for Apple Vision Pro

Integrate native support for Apple Vision Pro to leverage advanced visual capabilities.
  
- Data Collection and Synthetic Data Synthesis

Implement functionality for data collection and synthesis of simulated data to enhance training and operation.

# 🏆 Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

# ⚛️ Acknowledgments

This project draws inspiration and references from the following projects:

- [AnyTeleop](https://yzqin.github.io/anyteleop/)
- [DexRetargeting](https://github.com/dexsuite/dex-retargeting)
- [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision)
- [RoboCasa](https://github.com/robocasa/robocasa)
- [MimicGen](https://github.com/NVlabs/mimicgen)

# 📜 License

```text
Copyright 2023-2024 Herman Ye @Auromix
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. 
```
