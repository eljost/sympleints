module procedure {{ name }}  
  ! Note the swapped argument order BA instead of AB
  call {{ act_name }}(bxs, dbs, B, axs, das, A, R, res)
end procedure {{ name }}