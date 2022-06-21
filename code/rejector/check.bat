@REM Tests the main rejector and all other types of rejectors
python -m pyflakes .
python -m unittest -v
cd other_rejectors
python -m unittest -v
cd ..