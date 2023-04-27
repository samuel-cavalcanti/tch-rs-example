# tch-rs-example

Para executar a aplicação, clone este repositório

```bash
git clone https://github.com/samuel-cavalcanti/tch-rs-example
```

Baixe os dados do **NMIST**

```bash
./get_inputs.sh
```

execute a aplicação localmente

```bash
cargo r --release
```

Execute a aplicação na partição **gpu**

```bash
sbatch  sbatch run_on_superpc.sh
```
