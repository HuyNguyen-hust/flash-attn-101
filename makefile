run: build
	./bench
build:
	nvcc -o bench src/*