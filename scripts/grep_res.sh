#!/bin/bash
#DATA="Bump_2911 Cube_Coup_dt0 Flan_1565 ML_Geer dielFilterV3real Serena Geo_1438 af_shell10 ldoor Emilia_923 RM07R PFlow_742 ML_Laplace Fault_639 bundle_adj CoupCons3D pwtk PR02R bbmat hvdc2 scircuit windtunnel_evap3d hvdc1 StocF-1465 fullb fcondp2 HV15R"

DATA="dielFilterV3real Serena Geo_1438 af_shell10 ldoor StocF-1465 Emilia_923 RM07R PFlow_742 ML_Laplace Fault_639 bundle_adj CoupCons3D pwtk PR02R bbmat hvdc2 scircuit windtunnel_evap3d hvdc1 StocF-1465 fullb fcondp2"

queue="a2_20B"
#queue="a2_20Bg"
#queue="a1_10AE"
SOLVER=63

for DS in ${DATA};
do
    file="${queue}_solver_${SOLVER}_${DS}.log" 
    grep -e 'Dataset' -e 'Solve time' -e 'Residual Norm' -e 'Iterations' $file
    s=$(printf "%-30s" "*")
    echo "${s// /*}"
done

