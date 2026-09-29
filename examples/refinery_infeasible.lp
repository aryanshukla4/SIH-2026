\ INFEASIBLE VARIANT of refinery.lp: two sales orders added.
\ Refinery optimisation -- H. P. Williams, "Model Building in Mathematical
\ Programming" (5th ed., Wiley 2013), problem 12.6 / solution 13.6.
\ Two crudes -> distillation -> reforming + cracking -> blending into five
\ products. Profit in pence per barrel; published optimum 21,136,513 pence
\ (GBP 211,365.13) per day.
Maximize
 profit: 700 PMF + 600 RMF + 400 JF + 350 FO + 150 LBO
Subject To
 \ distillation yields per barrel of crude
 dist_LN: LN - 0.10 CR1 - 0.15 CR2 = 0
 dist_MN: MN - 0.20 CR1 - 0.25 CR2 = 0
 dist_HN: HN - 0.20 CR1 - 0.18 CR2 = 0
 dist_LO: LO - 0.12 CR1 - 0.08 CR2 = 0
 dist_HO: HO - 0.20 CR1 - 0.19 CR2 = 0
 dist_R:  R  - 0.13 CR1 - 0.12 CR2 = 0
 \ reforming, cracking, lube-oil conversion
 reform:   RG - 0.60 LNRG - 0.52 MNRG - 0.45 HNRG = 0
 crack_CO: CO - 0.68 LOCGO - 0.75 HOCGO = 0
 crack_CG: CG - 0.28 LOCGO - 0.20 HOCGO = 0
 lube:     LBO - 0.5 RLBO = 0
 \ where each intermediate goes (fuel oil is a fixed 10:4:3:1 recipe)
 use_LN: LN - LNRG - LNPMF - LNRMF = 0
 use_MN: MN - MNRG - MNPMF - MNRMF = 0
 use_HN: HN - HNRG - HNPMF - HNRMF = 0
 use_RG: RG - RGPMF - RGRMF = 0
 use_CG: CG - CGPMF - CGRMF = 0
 use_LO: LO - LOCGO - LOJF - 0.5555555555555556 FO = 0
 use_HO: HO - HOCGO - HOJF - 0.1666666666666667 FO = 0
 use_CO: CO - COJF - 0.2222222222222222 FO = 0
 use_R:  R - RLBO - RJF - 0.0555555555555556 FO = 0
 \ product blends
 blend_PMF: PMF - LNPMF - MNPMF - HNPMF - RGPMF - CGPMF = 0
 blend_RMF: RMF - LNRMF - MNRMF - HNRMF - RGRMF - CGRMF = 0
 blend_JF:  JF - LOJF - HOJF - COJF - RJF = 0
 \ quality specifications
 octane_premium: 90 LNPMF + 80 MNPMF + 70 HNPMF + 115 RGPMF + 105 CGPMF - 94 PMF >= 0
 octane_regular: 90 LNRMF + 80 MNRMF + 70 HNRMF + 115 RGRMF + 105 CGRMF - 84 RMF >= 0
 jet_vapour_pressure: LOJF + 0.6 HOJF + 1.5 COJF + 0.05 RJF - JF <= 0
 \ plant capacities and market rules
 distillation_cap: CR1 + CR2 <= 45000
 reforming_cap: LNRG + MNRG + HNRG <= 10000
 cracking_cap: LOCGO + HOCGO <= 8000
 premium_share: PMF - 0.4 RMF >= 0
 \ an order book the plant cannot meet
 order_premium: PMF >= 10000
 order_jet: JF >= 20000
Bounds
 CR1 <= 20000
 CR2 <= 30000
 500 <= LBO <= 1000
End
