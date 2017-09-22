# Face-Shazam
First Advanced Numerical Methods Project

## Getting Started
These instructions will install the development environment into your local machine.

### Prerequisites
1. Clone the repository
	```
	$ git clone https://github.com/juanmbellini/face_shazam.git
	```
2. Install Python and pip
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

### Setting path of subjects images
To set the path where the subjects images are, use the ```-tp``` or ```--subjects-path``` arguments. For example:
```
$ face_shazam -s ~/subjects
```
The default value is the directory ```./subjects/```.

### Setting images extension
To set the image extension, use the ```-ext``` or ```--image-extension``` arguments, including the extension
(without the dot). For example:
```
$ face_shazam -s ~/subjects -ext bpm
```

### Setting the recognition method
In order to set the recognition method (PCA or Kernel-PCA), use the ```-r``` or ```--recognize-method``` arguments,
together with the desired method (```pca```or ```kpca```).
For example:
```
$ face_shazam -s ~/subjects -ext bpm --recognize-method kpca
```

### Setting training percentage
To set the training percentage, use the ```-tp``` or ```--training-percentage``` arguments,
including the percentage to use. It must be a value between 0 (exclusive) and 1 (inclusive).
The training percentage indicates how many pictures (in percentage) will be used from each subject.
For example:
```
$ face_shazam -s ~/subjects -ext bpm -tp 0.6  # Will use only 6 images if the subject has 10
```
The default value is ```0.6```.

### Setting energy percentage
To set the energy percentage, use the ```-e``` or ```--energy-percentage``` arguments,
including the percentage to use. It must be a value between 0 (exclusive) and 1 (inclusive).
The energy percentage indicates how many eigen faces (in percentage of total sum of eigen values) 
will be used in the training process of the recognizer. Note that before truncating the list, 
the eigen vectors used to calculate the eigen faces are sorted according to their associated eigen value 
(being first in the list those with a bigger eigen value).
For example:
```
$ face_shazam -s ~/subjects -ext bpm -e 0.995
```
The default value is ```None``` (all eigen faces will be used).

### Setting kernel polynomial degree
To set the kernel polynomial degree, use the ```-kpd``` or ```--kernel-polynomial-degree``` arguments,
including the percentage to use. It must be an integer between 1 and 10.
This option should be used only with Kernel PCA recognition.
For example:
```
$ face_shazam -s ~/subjects -ext bpm -e 0.995 -kpd 3
```
The default value is ```2```.

### Setting subject to be recognized
To set the a subject to be recognized, use the ```-S``` or ```--subject``` arguments,
including the path to to the image of the subject.
If not set, the system will only display the score it achieved.
For example:
```
$ face_shazam -s ~/subjects -ext bpm -e 0.995 -kpd 3 --subject ./subjects/s3/8.pgm
```


## Authors
* [Juan Marcos Bellini](https://github.com/juanmbellini)
* [Tomás de Lucca](https://github.com/tomidelucca)
* [José Noriega](https://github.com/jcnoriega)
* [Agustín Scigliano](https://github.com/agustinscigliano)