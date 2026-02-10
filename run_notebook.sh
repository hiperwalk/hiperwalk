#!/bin/bash

# ================================
#  CONFIGURAÇÕES DO USUÁRIO
# ================================
INTEL_PY="/opt/intel/oneapi/intelpython/python3.12/bin/python"
INTEL_PY="python3"
KERNEL_NAME="hiperwalk"
DISPLAY_NAME="Python (hiperwalk)"
OMP_THREADS="${OMP_THREADS:-4}"    # valor padrão = 4
NOTEBOOK_IN="examples/coined/notebooks/coinedQW-on-HypercubeBD.ipynb"
NOTEBOOK_IN="examples/coined/notebooks/coinedQW-on-HypercubeBD.nbconvert.ipynb"
NOTEBOOK_OUT="saida.ipynb"
# ================================


echo "==➡️ Exportando variáveis MKL/OMP =="
export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"
echo "    OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "    MKL_NUM_THREADS=$MKL_NUM_THREADS"

echo ""
echo "==➡️ Verificando kernel '$KERNEL_NAME' =="

KERNEL_DIR="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME"

if [ ! -d "$KERNEL_DIR" ]; then
    echo "   ❌ Kernel não encontrado. Criando..."
    $INTEL_PY -m ipykernel install --user --name=$KERNEL_NAME --display-name="$DISPLAY_NAME"
else
    echo "   ✔️ Kernel já existe."
fi


echo ""
echo "==➡️ Executando notebook via nbconvert =="

comando="$INTEL_PY -m jupyter nbconvert \
    \"$NOTEBOOK_IN\" \
    --to notebook \
    --execute \
    --ExecutePreprocessor.kernel_name=$KERNEL_NAME \
    --TagFilterPreprocessor.enabled=True \
    --TagFilterPreprocessor.include=\"header\" \
    --output \"$NOTEBOOK_OUT\" "
echo $comando; eval $comando

STATUS=$?

echo ""
if [ $STATUS -eq 0 ]; then
    echo "==✔️ Execução concluída com sucesso! Arquivo gerado: $NOTEBOOK_OUT =="
else
    echo "==❌ Falha ao executar o notebook (status $STATUS) =="
fi

