\ Model Uncertain_qz
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  b[s,1_i] + b[2_o,g] + b[2_o,1_i] + b[1_o,2_i] + b[1_i,1_o] + b[2_i,2_o]
   + 0 R[1_i,0] + 0 R[1_i,1] + 0 R[1_i,2] + 0 R[1_i,3] + 0 R[1_i,4]
   + 0 R[1_i,5] + 0 R[1_i,6] + 0 R[1_i,7] + 0 R[1_i,8] + 0 R[1_i,9]
   + 0 R[2_i,0] + 0 R[2_i,1] + 0 R[2_i,2] + 0 R[2_i,3] + 0 R[2_i,4]
   + 0 R[2_i,5] + 0 R[2_i,6] + 0 R[2_i,7] + 0 R[2_i,8] + 0 R[2_i,9]
   + 0 X_bar[s,1_i] + 0 X_bar[2_o,g] + 0 X_bar[2_o,1_i] + 0 X_bar[1_o,2_i]
   + 0 X_bar[1_i,1_o] + 0 X_bar[2_i,2_o] + 0 Y_bar[s,1_i] + 0 Y_bar[2_o,g]
   + 0 Y_bar[2_o,1_i] + 0 Y_bar[1_o,2_i] + 0 Y_bar[1_i,1_o]
   + 0 Y_bar[2_i,2_o]
Subject To
 SOC_constraint_outside_quiet_zone_s1_i: a[s,1_i] - a_t[s,1_i]
   + 30 b[s,1_i] - 20 L[s,1_i] = 0
 Length_min_constraint_s1_i: b[s,1_i] - L[s,1_i] <= 0
 a_constraints_max_s_1_i: a[s,1_i] - 20 y[s,1_i] >= 0
 a_constraints_min_s_1_i: a[s,1_i] - 100 y[s,1_i] <= 0
 a_t_constraints_max_1_i_1_i: a_t[s,1_i] - 20 y[s,1_i] >= 0
 a_t_constraints_min_1_i_1_i: a_t[s,1_i] - 100 y[s,1_i] <= 0
 c_constraints_max_s_1_i: c[s,1_i] + 5 y[s,1_i] <= 0
 c_constraints_min_s_1_i: c[s,1_i] + 5 y[s,1_i] >= 0
 c_t_constraints_max_s_1_i: c_t[s,1_i] + 1.355005567935635 y[s,1_i] <= 0
 c_t_constraints_min_s_1_i: c_t[s,1_i] + 5.844994432064365 y[s,1_i] >= 0
 d_constraints_max_s_1_i: d[s,1_i] + 5 y[s,1_i] <= 0
 d_constraints_min_s_1_i: d[s,1_i] + 5 y[s,1_i] >= 0
 d_t_constraints_max_1_i_s: d_t[s,1_i] + 1.355005567935635 y[s,1_i] <= 0
 d_t_constraints_min_1_i_s: d_t[s,1_i] + 5.844994432064365 y[s,1_i] >= 0
 SOC_constraint_outside_quiet_zone_2_og: a[2_o,g] - a_t[2_o,g]
   + 30 b[2_o,g] - 20 L[2_o,g] = 0
 Length_min_constraint_2_og: b[2_o,g] - L[2_o,g] <= 0
 a_constraints_max_2_o_g: a[2_o,g] - 20 y[2_o,g] >= 0
 a_constraints_min_2_o_g: a[2_o,g] - 100 y[2_o,g] <= 0
 a_t_constraints_max_g_g: a_t[2_o,g] - 20 y[2_o,g] >= 0
 a_t_constraints_min_g_g: a_t[2_o,g] - 100 y[2_o,g] <= 0
 c_constraints_max_2_o_g: c[2_o,g] - 15.90743569830546 y[2_o,g] <= 0
 c_constraints_min_2_o_g: c[2_o,g] - 10.91609371345924 y[2_o,g] >= 0
 c_t_constraints_max_2_o_g: c_t[2_o,g] - 15 y[2_o,g] <= 0
 c_t_constraints_min_2_o_g: c_t[2_o,g] - 15 y[2_o,g] >= 0
 d_constraints_max_2_o_g: d[2_o,g] - 14 y[2_o,g] <= 0
 d_constraints_min_2_o_g: d[2_o,g] - 10.85553858101675 y[2_o,g] >= 0
 d_t_constraints_max_g_2_o: d_t[2_o,g] - 15 y[2_o,g] <= 0
 d_t_constraints_min_g_2_o: d_t[2_o,g] - 15 y[2_o,g] >= 0
 SOC_constraint_outside_quiet_zone_2_o1_i: a[2_o,1_i] - a_t[2_o,1_i]
   + 30 b[2_o,1_i] - 20 L[2_o,1_i] = 0
 Length_min_constraint_2_o1_i: b[2_o,1_i] - L[2_o,1_i] <= 0
 a_constraints_max_2_o_1_i: a[2_o,1_i] - 20 y[2_o,1_i] >= 0
 a_constraints_min_2_o_1_i: a[2_o,1_i] - 100 y[2_o,1_i] <= 0
 a_t_constraints_max_1_i_1_i: a_t[2_o,1_i] - 20 y[2_o,1_i] >= 0
 a_t_constraints_min_1_i_1_i: a_t[2_o,1_i] - 100 y[2_o,1_i] <= 0
 c_constraints_max_2_o_1_i: c[2_o,1_i] - 6 y[2_o,1_i] <= 0
 c_constraints_min_2_o_1_i: c[2_o,1_i] >= 0
 c_t_constraints_max_2_o_1_i: c_t[2_o,1_i] - 12 y[2_o,1_i] <= 0
 c_t_constraints_min_2_o_1_i: c_t[2_o,1_i] - 8 y[2_o,1_i] >= 0
 d_constraints_max_2_o_1_i: d[2_o,1_i] - 6 y[2_o,1_i] <= 0
 d_constraints_min_2_o_1_i: d[2_o,1_i] + 1.081967213114754 y[2_o,1_i] >= 0
 d_t_constraints_max_1_i_2_o: d_t[2_o,1_i] - 10.72131147540984 y[2_o,1_i]
   <= 0
 d_t_constraints_min_1_i_2_o: d_t[2_o,1_i] - 6 y[2_o,1_i] >= 0
 SOC_constraint_outside_quiet_zone_1_o2_i: a[1_o,2_i] - a_t[1_o,2_i]
   + 30 b[1_o,2_i] - 20 L[1_o,2_i] = 0
 Length_min_constraint_1_o2_i: b[1_o,2_i] - L[1_o,2_i] <= 0
 a_constraints_max_1_o_2_i: a[1_o,2_i] - 20 y[1_o,2_i] >= 0
 a_constraints_min_1_o_2_i: a[1_o,2_i] - 100 y[1_o,2_i] <= 0
 a_t_constraints_max_2_i_2_i: a_t[1_o,2_i] - 20 y[1_o,2_i] >= 0
 a_t_constraints_min_2_i_2_i: a_t[1_o,2_i] - 100 y[1_o,2_i] <= 0
 c_constraints_max_1_o_2_i: c[1_o,2_i] - 6 y[1_o,2_i] <= 0
 c_constraints_min_1_o_2_i: c[1_o,2_i] >= 0
 c_t_constraints_max_1_o_2_i: c_t[1_o,2_i] - 12 y[1_o,2_i] <= 0
 c_t_constraints_min_1_o_2_i: c_t[1_o,2_i] - 8 y[1_o,2_i] >= 0
 d_constraints_max_1_o_2_i: d[1_o,2_i] - 6 y[1_o,2_i] <= 0
 d_constraints_min_1_o_2_i: d[1_o,2_i] + 1.081967213114754 y[1_o,2_i] >= 0
 d_t_constraints_max_2_i_1_o: d_t[1_o,2_i] - 10.72131147540984 y[1_o,2_i]
   <= 0
 d_t_constraints_min_2_i_1_o: d_t[1_o,2_i] - 6 y[1_o,2_i] >= 0
 constr_z1_i: - y[1_i,1_o] + z[1_i,0] + z[1_i,1] + z[1_i,2] + z[1_i,3]
   + z[1_i,4] + z[1_i,5] + z[1_i,6] + z[1_i,7] + z[1_i,8] + z[1_i,9] = 0
 R57: - 30 z[1_i,0] + 1000 w[1_i,0] <= 0
 R58: - 30 z[1_i,1] + 4.902096352111123 w[1_i,1] <= 0
 R59: - 30 z[1_i,2] + 2.282397343340943 w[1_i,2] <= 0
 R60: - 30 z[1_i,3] + 1.406781858643888 w[1_i,3] <= 0
 R61: - 30 z[1_i,4] + 0.9660662339395073 w[1_i,4] <= 0
 R62: - 30 z[1_i,5] + 0.6981317007977319 w[1_i,5] <= 0
 R63: - 30 z[1_i,6] + 0.5151640100008957 w[1_i,6] <= 0
 R64: - 30 z[1_i,7] + 0.378761347706735 w[1_i,7] <= 0
 R65: - 30 z[1_i,8] + 0.2681254619972019 w[1_i,8] <= 0
 R66: - 30 z[1_i,9] + 0.1670469673319492 w[1_i,9] <= 0
 R67: L[1_i,1_o] - 12 z[1_i,0] - 11.93984924527944 z[1_i,1]
   - 11.75755076535926 z[1_i,2] - 11.44727041700335 z[1_i,3]
   - 10.99818166789402 z[1_i,4] - 10.39230484541326 z[1_i,5] - 9.6 z[1_i,6]
   - 8.56971411425142 z[1_i,7] - 7.2 z[1_i,8] - 5.230678732248811 z[1_i,9]
   = 0
 R68: - 0.1666666666666667 L[1_i,1_o] + y[1_i,1_o] >= 0
 R69: b[1_i,1_o] - 12 w[1_i,0] - 11.93984924527944 w[1_i,1]
   - 11.75755076535926 w[1_i,2] - 11.44727041700335 w[1_i,3]
   - 10.99818166789402 w[1_i,4] - 10.39230484541326 w[1_i,5] - 9.6 w[1_i,6]
   - 8.56971411425142 w[1_i,7] - 7.2 w[1_i,8] - 5.230678732248811 w[1_i,9]
   = 0
 R70: - z[1_i,0] + w[1_i,0] <= 0
 R71: - z[1_i,1] + w[1_i,1] <= 0
 R72: - z[1_i,2] + w[1_i,2] <= 0
 R73: - z[1_i,3] + w[1_i,3] <= 0
 R74: - z[1_i,4] + w[1_i,4] <= 0
 R75: - z[1_i,5] + w[1_i,5] <= 0
 R76: - z[1_i,6] + w[1_i,6] <= 0
 R77: - z[1_i,7] + w[1_i,7] <= 0
 R78: - z[1_i,8] + w[1_i,8] <= 0
 R79: - z[1_i,9] + w[1_i,9] <= 0
 R80: w[1_i,0] >= 0
 R81: w[1_i,1] >= 0
 R82: w[1_i,2] >= 0
 R83: w[1_i,3] >= 0
 R84: w[1_i,4] >= 0
 R85: w[1_i,5] >= 0
 R86: w[1_i,6] >= 0
 R87: w[1_i,7] >= 0
 R88: w[1_i,8] >= 0
 R89: w[1_i,9] >= 0
 constr_z2_i: - y[2_i,2_o] + z[2_i,0] + z[2_i,1] + z[2_i,2] + z[2_i,3]
   + z[2_i,4] + z[2_i,5] + z[2_i,6] + z[2_i,7] + z[2_i,8] + z[2_i,9] = 0
 R91: - 30 z[2_i,0] + 1000 w[2_i,0] <= 0
 R92: - 30 z[2_i,1] + 7.353144528166684 w[2_i,1] <= 0
 R93: - 30 z[2_i,2] + 3.423596015011415 w[2_i,2] <= 0
 R94: - 30 z[2_i,3] + 2.110172787965831 w[2_i,3] <= 0
 R95: - 30 z[2_i,4] + 1.44909935090926 w[2_i,4] <= 0
 R96: - 30 z[2_i,5] + 1.047197551196598 w[2_i,5] <= 0
 R97: - 30 z[2_i,6] + 0.7727460150013433 w[2_i,6] <= 0
 R98: - 30 z[2_i,7] + 0.5681420215601024 w[2_i,7] <= 0
 R99: - 30 z[2_i,8] + 0.4021881929958027 w[2_i,8] <= 0
 R100: - 30 z[2_i,9] + 0.2505704509979235 w[2_i,9] <= 0
 R101: L[2_i,2_o] - 8 z[2_i,0] - 7.95989949685296 z[2_i,1]
   - 7.83836717690617 z[2_i,2] - 7.631513611335565 z[2_i,3]
   - 7.332121111929344 z[2_i,4] - 6.928203230275509 z[2_i,5] - 6.4 z[2_i,6]
   - 5.71314274283428 z[2_i,7] - 4.8 z[2_i,8] - 3.487119154832538 z[2_i,9]
   = 0
 R102: - 0.25 L[2_i,2_o] + y[2_i,2_o] >= 0
 R103: b[2_i,2_o] - 8 w[2_i,0] - 7.95989949685296 w[2_i,1]
   - 7.83836717690617 w[2_i,2] - 7.631513611335565 w[2_i,3]
   - 7.332121111929344 w[2_i,4] - 6.928203230275509 w[2_i,5] - 6.4 w[2_i,6]
   - 5.71314274283428 w[2_i,7] - 4.8 w[2_i,8] - 3.487119154832538 w[2_i,9]
   = 0
 R104: - z[2_i,0] + w[2_i,0] <= 0
 R105: - z[2_i,1] + w[2_i,1] <= 0
 R106: - z[2_i,2] + w[2_i,2] <= 0
 R107: - z[2_i,3] + w[2_i,3] <= 0
 R108: - z[2_i,4] + w[2_i,4] <= 0
 R109: - z[2_i,5] + w[2_i,5] <= 0
 R110: - z[2_i,6] + w[2_i,6] <= 0
 R111: - z[2_i,7] + w[2_i,7] <= 0
 R112: - z[2_i,8] + w[2_i,8] <= 0
 R113: - z[2_i,9] + w[2_i,9] <= 0
 R114: w[2_i,0] >= 0
 R115: w[2_i,1] >= 0
 R116: w[2_i,2] >= 0
 R117: w[2_i,3] >= 0
 R118: w[2_i,4] >= 0
 R119: w[2_i,5] >= 0
 R120: w[2_i,6] >= 0
 R121: w[2_i,7] >= 0
 R122: w[2_i,8] >= 0
 R123: w[2_i,9] >= 0
 Sink_max_SOC_constraint: a[s,1_i] = 100
 Sink_max_SOC_constraint: a_t[2_o,g] <= 100
 Sink_min_SOC_constraint: a_t[2_o,g] >= 20
 R127: y[s,1_i] = 1
 R128: y[2_o,g] = 1
 R129: - y[s,1_i] - y[2_o,1_i] + y[1_i,1_o] = 0
 R130: y[2_o,g] + y[2_o,1_i] - y[2_i,2_o] = 0
 R131: y[1_o,2_i] - y[1_i,1_o] = 0
 R132: - y[1_o,2_i] + y[2_i,2_o] = 0
 R133: y[1_i,1_o] <= 1
 R134: y[2_o,g] + y[2_o,1_i] <= 1
 R135: y[1_o,2_i] <= 1
 R136: y[2_i,2_o] <= 1
 R137: y[s,1_i] + y[2_o,1_i] <= 1
 R138: y[2_i,2_o] <= 1
 R139: y[1_i,1_o] <= 1
 R140: y[1_o,2_i] <= 1
 R141: a[1_i,1_o] - a_t[s,1_i] - a_t[2_o,1_i] = 0
 R142: a[2_o,g] + a[2_o,1_i] - a_t[2_i,2_o] = 0
 R143: a[1_o,2_i] - a_t[1_i,1_o] = 0
 R144: a[2_i,2_o] - a_t[1_o,2_i] = 0
 R145: c[1_i,1_o] - c_t[s,1_i] - c_t[2_o,1_i] = 0
 R146: c[2_o,g] + c[2_o,1_i] - c_t[2_i,2_o] = 0
 R147: c[1_o,2_i] - c_t[1_i,1_o] = 0
 R148: c[2_i,2_o] - c_t[1_o,2_i] = 0
 SOCP_constr_s1_i: [ - L[s,1_i] ^2 + c[s,1_i] ^2 - 2 c[s,1_i] * c_t[s,1_i]
   + c_t[s,1_i] ^2 + d[s,1_i] ^2 - 2 d[s,1_i] * d_t[s,1_i] + d_t[s,1_i] ^2
   ] <= 0
 constr_c_ds: [ c[s,1_i] ^2 + 10 c[s,1_i] * y[s,1_i] + d[s,1_i] ^2
   + 10 d[s,1_i] * y[s,1_i] + 50 y[s,1_i] ^2 ] <= 0
 constr_c_d_t1_i: [ c_t[s,1_i] ^2 + d_t[s,1_i] ^2 - 36 y[s,1_i] ^2 ] <= 0
 SOCP_constr_2_og: [ - L[2_o,g] ^2 + c[2_o,g] ^2 - 2 c[2_o,g] * c_t[2_o,g]
   + c_t[2_o,g] ^2 + d[2_o,g] ^2 - 2 d[2_o,g] * d_t[2_o,g] + d_t[2_o,g] ^2
   ] <= 0
 constr_c_d2_o: [ c[2_o,g] ^2 - 24 c[2_o,g] * y[2_o,g] + d[2_o,g] ^2
   - 20 d[2_o,g] * y[2_o,g] + 228 y[2_o,g] ^2 ] <= 0
 constr_c_d_tg: [ c_t[2_o,g] ^2 - 30 c_t[2_o,g] * y[2_o,g] + d_t[2_o,g] ^2
   - 30 d_t[2_o,g] * y[2_o,g] + 450 y[2_o,g] ^2 ] <= 0
 SOCP_constr_2_o1_i: [ - L[2_o,1_i] ^2 + c[2_o,1_i] ^2
   - 2 c[2_o,1_i] * c_t[2_o,1_i] + c_t[2_o,1_i] ^2 + d[2_o,1_i] ^2
   - 2 d[2_o,1_i] * d_t[2_o,1_i] + d_t[2_o,1_i] ^2 ] <= 0
 constr_c_d2_o: [ c[2_o,1_i] ^2 - 24 c[2_o,1_i] * y[2_o,1_i]
   + d[2_o,1_i] ^2 - 20 d[2_o,1_i] * y[2_o,1_i] + 228 y[2_o,1_i] ^2 ] <= 0
 constr_c_d_t1_i: [ c_t[2_o,1_i] ^2 + d_t[2_o,1_i] ^2 - 36 y[2_o,1_i] ^2 ]
   <= 0
 SOCP_constr_1_o2_i: [ - L[1_o,2_i] ^2 + c[1_o,2_i] ^2
   - 2 c[1_o,2_i] * c_t[1_o,2_i] + c_t[1_o,2_i] ^2 + d[1_o,2_i] ^2
   - 2 d[1_o,2_i] * d_t[1_o,2_i] + d_t[1_o,2_i] ^2 ] <= 0
 constr_c_d1_o: [ c[1_o,2_i] ^2 + d[1_o,2_i] ^2 - 36 y[1_o,2_i] ^2 ] <= 0
 constr_c_d_t2_i: [ c_t[1_o,2_i] ^2 - 24 c_t[1_o,2_i] * y[1_o,2_i]
   + d_t[1_o,2_i] ^2 - 20 d_t[1_o,2_i] * y[1_o,2_i] + 228 y[1_o,2_i] ^2 ]
   <= 0
 constr_length_internal1_i: - 144 z[1_i,0] - 142.56 z[1_i,1]
   - 138.24 z[1_i,2] - 131.04 z[1_i,3] - 120.96 z[1_i,4] - 108 z[1_i,5]
   - 92.16 z[1_i,6] - 73.44 z[1_i,7] - 51.84 z[1_i,8] - 27.36 z[1_i,9] + [
   c[1_i,1_o] ^2 - 2 c[1_i,1_o] * c_t[1_i,1_o] + c_t[1_i,1_o] ^2
   + d[1_i,1_o] ^2 - 2 d[1_i,1_o] * d_t[1_i,1_o] + d_t[1_i,1_o] ^2 ] <= 0
 constr_c_d1_i: [ c[1_i,1_o] ^2 + d[1_i,1_o] ^2 - 36 y[1_i,1_o] ^2 ] <= 0
 constr_c_d_t1_o: [ c_t[1_i,1_o] ^2 + d_t[1_i,1_o] ^2 - 36 y[1_i,1_o] ^2 ]
   <= 0
 qc15: a[1_i,1_o] - a_t[1_i,1_o] - 20 L[1_i,1_o] + [
   30 L[1_i,1_o] * w[1_i,0] + 30 L[1_i,1_o] * w[1_i,1]
   + 30 L[1_i,1_o] * w[1_i,2] + 30 L[1_i,1_o] * w[1_i,3]
   + 30 L[1_i,1_o] * w[1_i,4] + 30 L[1_i,1_o] * w[1_i,5]
   + 30 L[1_i,1_o] * w[1_i,6] + 30 L[1_i,1_o] * w[1_i,7]
   + 30 L[1_i,1_o] * w[1_i,8] + 30 L[1_i,1_o] * w[1_i,9] ] = 0
 constr_length_internal2_i: - 64 z[2_i,0] - 63.36 z[2_i,1] - 61.44 z[2_i,2]
   - 58.24 z[2_i,3] - 53.76 z[2_i,4] - 48 z[2_i,5] - 40.96 z[2_i,6]
   - 32.64 z[2_i,7] - 23.04 z[2_i,8] - 12.16 z[2_i,9] + [ c[2_i,2_o] ^2
   - 2 c[2_i,2_o] * c_t[2_i,2_o] + c_t[2_i,2_o] ^2 + d[2_i,2_o] ^2
   - 2 d[2_i,2_o] * d_t[2_i,2_o] + d_t[2_i,2_o] ^2 ] <= 0
 constr_c_d2_i: [ c[2_i,2_o] ^2 - 24 c[2_i,2_o] * y[2_i,2_o]
   + d[2_i,2_o] ^2 - 20 d[2_i,2_o] * y[2_i,2_o] + 228 y[2_i,2_o] ^2 ] <= 0
 constr_c_d_t2_o: [ c_t[2_i,2_o] ^2 - 24 c_t[2_i,2_o] * y[2_i,2_o]
   + d_t[2_i,2_o] ^2 - 20 d_t[2_i,2_o] * y[2_i,2_o] + 228 y[2_i,2_o] ^2 ]
   <= 0
 qc19: a[2_i,2_o] - a_t[2_i,2_o] - 20 L[2_i,2_o] + [
   30 L[2_i,2_o] * w[2_i,0] + 30 L[2_i,2_o] * w[2_i,1]
   + 30 L[2_i,2_o] * w[2_i,2] + 30 L[2_i,2_o] * w[2_i,3]
   + 30 L[2_i,2_o] * w[2_i,4] + 30 L[2_i,2_o] * w[2_i,5]
   + 30 L[2_i,2_o] * w[2_i,6] + 30 L[2_i,2_o] * w[2_i,7]
   + 30 L[2_i,2_o] * w[2_i,8] + 30 L[2_i,2_o] * w[2_i,9] ] = 0
Bounds
 c[s,1_i] free
 c[2_o,g] free
 c[2_o,1_i] free
 c[1_o,2_i] free
 c[1_i,1_o] free
 c[2_i,2_o] free
 c_t[s,1_i] free
 c_t[2_o,g] free
 c_t[2_o,1_i] free
 c_t[1_o,2_i] free
 c_t[1_i,1_o] free
 c_t[2_i,2_o] free
 d[s,1_i] free
 d[2_o,g] free
 d[2_o,1_i] free
 d[1_o,2_i] free
 d[1_i,1_o] free
 d[2_i,2_o] free
 d_t[s,1_i] free
 d_t[2_o,g] free
 d_t[2_o,1_i] free
 d_t[1_o,2_i] free
 d_t[1_i,1_o] free
 d_t[2_i,2_o] free
Binaries
 y[s,1_i] y[2_o,g] y[2_o,1_i] y[1_o,2_i] y[1_i,1_o] y[2_i,2_o] z[1_i,0]
 z[1_i,1] z[1_i,2] z[1_i,3] z[1_i,4] z[1_i,5] z[1_i,6] z[1_i,7] z[1_i,8]
 z[1_i,9] z[2_i,0] z[2_i,1] z[2_i,2] z[2_i,3] z[2_i,4] z[2_i,5] z[2_i,6]
 z[2_i,7] z[2_i,8] z[2_i,9] w[1_i,0] w[1_i,1] w[1_i,2] w[1_i,3] w[1_i,4]
 w[1_i,5] w[1_i,6] w[1_i,7] w[1_i,8] w[1_i,9] w[2_i,0] w[2_i,1] w[2_i,2]
 w[2_i,3] w[2_i,4] w[2_i,5] w[2_i,6] w[2_i,7] w[2_i,8] w[2_i,9]
End
