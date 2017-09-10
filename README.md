# Face-Shazam
First Advanced Numerical Methods Project

## Getting Started
These instructions will install the development environment into your local machine.

### Prerequisites
1. Clone the repository or download the source code

	```
	$ git clone https://github.com/juanmbellini/face_shazam.git
	```
	or
	```
	$ wget https://github.com/juanmbellini/face_shazam/archive/master.zip
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

### Build
1. Move to the new virtual environment
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

## Usage

## Authors
* [Juan Marcos Bellini](https://github.com/juanmbellini)
* [Tomás de Lucca](https://github.com/tomidelucca)
* [José Noriega](https://github.com/jcnoriega)
* [Agustín Scigliano](https://github.com/agustinscigliano)