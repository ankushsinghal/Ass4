if [ $# -ne 1 ]; then
	echo "Error: Script expects only one command line argument..."
fi

base64 -d $1 > {1234567abcd}.tar.gz
tar -xzvf {1234567abcd}.tar.gz
rm {1234567abcd}.tar.gz