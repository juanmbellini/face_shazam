# Face-Shazam
First Advanced Numerical Methods Project

## Getting Started
These instructions will install the development environment into your local machine.

### Prerequisites
1. Clone the repository
	```
	$ git clone https://github.com/juanmbellini/face_shazam.git
	```
2. Install Python and Python Package Index (pip)
	#### MacOS
	A. Install packages
	```
	$ brew install python
	```
	B. Update the ```PATH``` variable to use the Homebrew's python packages
	```
	$ echo 'export PATH="/usr/local/opt/python/libexec/bin:$PATH" # Use Homebrew python' >> ~/.bash_profile
	$ source ~/.bash_profile
	
	```  
	#### Ubuntu
	```
	$ sudo apt-get install python python-pip
	```
3. Install [Virtualenv](https://virtualenv.pypa.io/en/latest/) 
	and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/)
	```
	$ pip install virtualenv virtualenvwrapper
	$ echo 'source /usr/local/bin/virtualenvwrapper.sh # Virtualenv/VirtualenvWrapper' >> ~/.bash_profile
	$ source ~/.bash_profile
	```
4. Create a virtual environment for the project
	```
	$ mkvirtualenv face_shazam
	```
	**Note:** This will install ```setuptools```, ```pip``` and ```wheel``` modules in the new virtual environment.

### Build
1. Move to the new virtual environment and change working directory to project's root
	```
	$ workon face_shazam
	```
	**Note:** To leave the virtual environment, execute 
    ```
    $ deactivate
    ```
2. Install dependencies
	```
	$ pip install -r requirements.txt
	```
3. Install module
	```
	$ python setup.py clean --all install
	```

## Usage
The application can be run executing the ``face_shazam`` command. 
The following sections will explain the different options and arguments that can be used.

### Displaying usage message
You can display the usage message using the ```-h``` or ```--help``` arguments. For example:
```
$ face_shazam --help
```

### Displaying version number
You can check the version of the module using the ```-V``` or ```--version``` arguments. For example:
```
$ face_shazam -V
```

### Logging verbosity

#### Logging levels
There are three levels of logging verbosity: 
* **Normal**
* **Verbose** 
* **Very Verbose**.

Normal verbosity logging will log **WARNING**, **ERROR** and **CRITICAL** messages.
Verbose logging will log what Normal logging logs, and **INFO** messages.
Very Verbose logging is the same as Verbose logging, adding **DEBUG** messages.

#### Selecting a logging level
##### Normal Verbosity Logging
To use **Normal** verbosity logging just execute the command. **Normal** verbosity logging is the default
##### Verbose Logging
To use **Verbose** logging, add the ```-v``` or ```--verbose``` arguments. For example:
```
$ face_shazam --verbose
```
##### Very Verbose Logging
To use **Very Verbose** logging, add the ```-vv``` or ```--very-verbose``` arguments. For example:
```
$ face_shazam -vv
```

## Authors
* [Juan Marcos Bellini](https://github.com/juanmbellini)
* [Tomás de Lucca](https://github.com/tomidelucca)
* [José Noriega](https://github.com/jcnoriega)
* [Agustín Scigliano](https://github.com/agustinscigliano)