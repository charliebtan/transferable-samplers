sequences=(
    AC
    AT
    ET
    GN
    GP
    HT
    IM
    KG
    KQ
    KS
    LW
    NF
    NY
    RL
    RV
    TD
    SAEL
    RYDT
    CSFQ
    FALS
    CSGS
    LPEM
    LYVI
    AYTG
    VCVS
    AAEW
    FKVP
    NQFM
    DTDL
    CTSA
    ANYT
    VTST
    AWKC
    RGSP
    AVEK
    FIYG
    VLSM
    QADY
    DQAL
    TFFL
    FIGE
    KKQF
    SLTC
    ITQD
    DFKS
    QDED
    PGESTAES
    NKEKFFQH
    MYGRNCYM
    IDHRQLKW
    HWHSLICK
    NPCLCYML
    MRDPVLFA
    DDRDTEQT
    YFPHAGYT
    ISKCKNGE
    KRRGFFLE
    CLCCGQWN
    GNDLVTVI
    EKYYWMQT
    FWRVDHDM
    DGVAHALS
    PLFHVMYV
    SQQKVAFE
    IFGWVYTG
    CGSWHKQR
    WTYAFAHS
    MWNSTEMI
    PYIRNCVE
    ANKSMIEA
    MAPQTIAT
    SPHKMRLC
    VWIPVIDT
    NHQYGSDP
    PPWRECNN
)

maxiters=(0 10 100 1000)

for seq in "${sequences[@]}"; do
    for maxiter in "${maxiters[@]}"; do
        python src/train.py -m \
            experiment=evaluation/tarflow_up_to_8aa \
            logger=wandb \
            model.eval_seq_name="$seq" \
            +model.dont_fix_symmetry=True \
            +model.energy_maxiter="$maxiter" \
            +model.sample_set=unisim
    done
done