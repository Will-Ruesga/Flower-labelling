# Set up WSL & SAM3 with Pip

### 1 - Install WSL2

To isntall wsl2 open Windows PowerShell and run the following commands
and follow the installation instructions
```
	wsl --install 
```

Check if it worked by using
```
	wsl --version
```

If it outputs something similar to this it is correct and WSL2 hs been correctly installed.
```
	WSL version: 2.6.1.0
	Kernel version: 6.6.87.2-1
	WSLg version: 1.0.66
	MSRDC version: 1.2.6353
	Direct3D version: 1.611.1-81528511
	DXCore version: 10.0.26100.1-240331-1435.ge-release
	Windows version: 10.0.26100.4770
```
---

### 2 - Setup WSL2

Open Windows Terminal app and select a new Ubuntu window
(If desired Ubuntu can be set up as the default in the Terminal app)
Run the following commands to install the essentials

```
	sudo apt update && sudo apt upgrade -y
	sudo apt install python3-pip python3-venv python3-full -y
```

---

### 3 - Create virtual environment and install pytorch

Create the virtual environment in WSL by running

```
	python3 -m venv <ENV_NAME>
```

Active the environment

```
	source <ENV_NAME>/bin/activate
```

Should appear in the left like this:
(<ENV_NAME>) user@User:~/

Download and install pytorch in the environment
Go to: https://pytorch.org/get-started/locally/ to dowload with the following settings and run the command it says

+ **Pytorch build**: Stable
+ **OS**: Windows (becuase you are working with WSL)
+ **Package**: pip
+ **Language**: Python
+ **Compute platform**: (Select the CUDA version you have installed)

Check Pytorch is working and it is using the GPU
```
	python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
---
	
### 4 - SAM3

Dowload the model SAM3 in pip first clone the github and enter 
```
	git clone https://github.com/facebookresearch/sam3.git && cd sam3
```

Once in the SAM3 repository, get only the sam3 folder.
The default structure should look like this:

```
labelling-repo/
├─ sam3/                    # -- SAM3 repository [Not included] -- #
│   └─sam3/					# Required
│   └─ Other files			# Unnecessary
```

The only files used are inside sam3/sam3 therefore remvoe the rest 
and put the subfolder as the main sam3 folder like so:

```
labelling-repo/
├─ sam3/
│   └─ files			# Here there are files like model_builder.py and some folders
```

---

### 5 - Model checkpoints
SAM3 Model checkpoints should be resolved or have to be asked and downloaded from Huggingface