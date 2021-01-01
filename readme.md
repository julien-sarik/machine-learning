# build image
```
docker build -t sklearn:latest .
```

# run image
````
docker run -it --rm --name sklearn `
-v "C:\Users\julsarik\projects\machine-learning\data:/data" `
-v "C:\Users\julsarik\projects\machine-learning\scripts:/scripts" `
sklearn bash
```