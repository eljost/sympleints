module procedure {{ name }}
  ! Note the swapped argument order BADC instead of ABCD
  !                                                              11 22   12 12
  ! Currently, this is only suitable for Schwarz-type integrals (ij|ij)/<ii|jj>.
  call {{ act_name }}(bxs, dbs, B, axs, das, A, dxs, dds, D, cxs, dcs, C, res)
end procedure {{ name }}
