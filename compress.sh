if [ $# -eq 0 ]; then
	echo "Error: No input file provided..."
fi

tar czvf kaggle_assignment.tar.gz $@
base64 kaggle_assignment.tar.gz > kaggle_assignment.tar.gz.b64
rm kaggle_assignment.tar.gz