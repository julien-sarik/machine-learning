# build image
```
docker build -t pandas:latest .
```

# run image
````
docker run -it --rm --name pandas -d ^
-v "C:\Users\julsarik\projects\machine-learning\data:/data" ^
-v "C:\Users\julsarik\projects\machine-learning\scripts:/scripts" ^
pandas sh
```