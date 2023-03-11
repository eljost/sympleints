program test_ovlp
    use iso_fortran_env, only: int64, real64
    use mod_{{ key }}
    {% if need_boys %}use mod_boys{% endif %}

    implicit none

    integer :: lmax, lauxmax
    integer :: nprims
    integer :: niters
    real(kind=real64), allocatable :: ax(:), da(:), bx(:), db(:), cx(:), dc(:)
    real(kind=real64) :: A(3), B(3), C(3), R(3)
    integer :: la, lb, n
    real(kind=real64), allocatable :: res(:, :, :)
    integer(int64) :: count_start, count_end, count_rate
    real(kind=real64) :: tot_time, iter_time
    logical :: is_spherical
    integer :: res_shape(2)
    integer :: ncomponents
    character(len=10) :: key

    ! User input
    lmax = {{ lmax }}
    lauxmax = -1
    nprims = {{ nprims }}
    niters = {{ niters }}
    is_spherical = {{ is_spherical }}
    ncomponents = {{ ncomponents }}
    key = "{{ key }}"
    ! User input end

    allocate(ax(nprims))
    allocate(da(nprims))
    allocate(bx(nprims))
    allocate(db(nprims))
    allocate(cx(nprims))
    allocate(dc(nprims))

    ! Initialize integral procedures
    call {{ key }}_init()
    {% if need_boys %}call boys_init(){% endif %}

    ! Initialize timer
    call system_clock(count_rate=count_rate)
    tot_time = 0.0
    ! Reference center
    call random_number(R)

    do la = 0, lmax
        call random_number(ax)
        call random_number(da)
        call random_number(A)
        do lb = 0, lmax
            call random_number(bx)
            call random_number(db)
            call random_number(B)

            ! Allocate results array with correct shape, depending on la and lb
            if (is_spherical) then
                res_shape = (/ 2*la + 1, 2*lb +1 /)
            else
                res_shape = (/ (la+2)*(la+1)/2, (lb+2)*(lb+1)/2 /)
            end if
            allocate(res(ncomponents, res_shape(1), res_shape(2)))
            tot_time = 0.0

            ! Call actual integration procedure
            call system_clock(count=count_start)
            do n = 1, niters
                call {{ key }}(la, lb, ax, da, A, bx, db, B, R, res)
            end do
            call system_clock(count=count_end)
            tot_time = tot_time + (real(count_end - count_start, kind=real64) / count_rate)
            iter_time = tot_time / niters
            write (*, "(I4, I4, F16.6, ES16.6)") &
                la, lb, tot_time, iter_time
            deallocate(res)
        end do
    end do

end program test_ovlp
