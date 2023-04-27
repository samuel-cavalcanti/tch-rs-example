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

## sem Cuda

```bash
cargo r --release
```

## com Cuda

```bash
# compilando com cuda versão 11.7
export TORCH_CUDA_VERSION=cu117
cargo r --release
```

## Submetendo o Job
 
Execute a aplicação na partição **gpu**

```bash
sbatch  sbatch run_on_superpc.sh
```
