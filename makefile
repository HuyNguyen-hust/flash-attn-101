run: build
	./bench
build:
	nvcc -Iinclude -o bench src/*