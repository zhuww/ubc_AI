!f2py -c -m nnopt --f90flags="-fopenmp -lm " --opt="-O3 " -lgomp --fcompiler=gnu95 nn_optimize.f90
subroutine sigmoid2d(z,nr,nc)
  use omp_lib
  implicit none
  real*8 z(nr,nc)
  integer nr, nc
  integer :: i, j, num_threads
  !f2py intent(hide) :: nr, nc
  !f2py intent(in,out) :: z
  
!$OMP PARALLEL DO PRIVATE(j),shared(z)
  do i=1,nr
!     if (j == 1) then
        ! The if clause can be removed for serious use.
        ! It is here for debugging only.
!        num_threads = OMP_get_num_threads()
!        print *, 'num_threads running:', num_threads
!     end if
     do j=1,nc
        z(i,j) = 1.d0/(1.d0 + exp(-1.d0*z(i,j)))
     end do
  end do
!$OMP END PARALLEL DO

end subroutine sigmoid2d

  
subroutine matmult(A,B,C,nra,nca,nrb,ncb)
  use omp_lib
  implicit none
  real*8 A(nra,nca), B(nrb,ncb), C(nra,ncb)
  integer nra, nca, nrb, ncb, chunk
  integer :: i, j, k
  !f2py intent(in) :: A, B
  !f2py intent(hide) :: C, nra, nca, nrb, ncb
  !f2py intent(out) :: C
  !f2py depends(nra,nca) :: A
  !f2py depends(nrb,ncb) :: B
  !f2py depends(nra,ncb) :: C
  
  
  !     Set loop iteration chunk size 
  chunk = 10

!     Spawn a parallel region explicitly scoping all variables
!$OMP PARALLEL SHARED(A,B,C,chunk) PRIVATE(i,j,k)
!$OMP DO SCHEDULE(STATIC, chunk)
  do i=1, nra
     do j=1,ncb
        C(i,j) = 0.
     end do
  end do


!     Do matrix multiply sharing iterations on outer loop
!C     Display who does which iterations for demonstration purposes
  !$OMP DO SCHEDULE(STATIC, chunk)
  do i=1, nra
     do j=1, ncb
        do k=1, nca
           C(i,j) = C(i,j) + A(i,k) * B(k,j)
        end do
     end do
  end do
  !     End of parallel region 
  !$OMP END PARALLEL

end subroutine matmult
