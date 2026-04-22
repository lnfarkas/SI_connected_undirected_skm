from reader_npz_to_csv_curves_edges_tripoint import npz_to_csv
simID = "SIsimUNDIRECTED20260422104108"
Nc = "20"
Np = "2"
npz_to_csv(
    f"/home/lnf/Desktop/00_sim_SI/{simID}/N20_k2/Curves/curves_instanceNo0000_Nprocesses{Np}_N20_Nconnected{Nc}_k2.0_{simID}.npz",
    f"/home/lnf/Desktop/00_sim_SI/{simID}/N20_k2/Curves/curves_instanceNo0000_Nprocesses{Np}_N20_Nconnected{Nc}_k2.0_{simID}.csv"
)
