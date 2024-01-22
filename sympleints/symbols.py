from sympleints import get_center, get_map


# Reference center/multipole origin
center_R = get_center("R")
R, R_map = get_map("R", center_R)

# Coordinates at which a function is evaluated
center_r = get_center("r")
r, r_map = get_map("r", center_r)

# Center-of-charge coordinate
center_P = get_center("P")
P, P_map = get_map("P", center_P)


center_rP = get_center("rP")
rP, rP_map = get_map("rP", center_rP)


center_rP2 = get_center("rP2")
rP2, rP2_map = get_map("rP2", center_rP2)
