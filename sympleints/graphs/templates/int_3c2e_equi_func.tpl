module procedure {{ name }}
  ! Note the swapped argument order BAC instead of ABC
  call {{ act_name }}(bxs, dbs, B, axs, das, A, cxs, dcs, C, res)
end procedure {{ name }}
