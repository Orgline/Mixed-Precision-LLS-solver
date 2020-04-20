# Mixed-Precision-LLS-solver

Run RGSQRF
```
./lls 1024 1024 1 0 1
```

Run sgeqrf
```
./lls 1024 1024 2 0 
```

Run dgeqrf
```
./lls 1024 1024 3 0
```

Run RGSQRF LRA
```
./lls 1024 1024 4 0 64
```

Run SGEQRF LRA
```
./lls 1024 1024 5 0 64
```

Run cgls
```
./cgls 1024 1024
```
