#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <mpi.h>

int main( int argc, char* argv[] ) {

    int dim1 = 6;
    int dim2 = 4;
    int dim3 = 1;
    int dim4 = 2;

    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
    std::cout << "Testing Hilbert Layout\n"; 

    Kokkos::View<double****, Kokkos::LayoutHilbertRight, Kokkos::HostSpace> HilbertArray( "Hilbert", dim1, dim2, dim3, dim4 );
    Kokkos::View<double****> RegularArray( "Regular", dim1, dim2, dim3, dim4 );

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            for (int k = 0; k < dim3; k++ ) {
                for ( int l = 0; l < dim4; l++ ) {
                    HilbertArray( i, j, k, l ) = i + dim1 * ( j + dim2 * ( k + ( dim3 ) * l ) );
                    RegularArray( i, j, k, l ) = i + dim1 * ( j + dim2 * ( k + ( dim3 ) * l ) );
                }
            }
        }
    }

    std::cout << "Hilbert Array Var 1\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << HilbertArray( i, j, 0, 0 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Hilbert Array Var 2\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << HilbertArray( i, j, 0, 1 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Regular Array Var 1\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << RegularArray( i, j, 0, 0 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Hilbert Array Var 2\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << RegularArray( i, j, 0, 1 ) << " ";
        }
        std::cout << "\n";
    }

    Cajita::IndexSpace<4> space;
    space = Cajita::IndexSpace<4>( { 0, 0, 0, 0 }, { dim1, 2, dim3, dim4 } );

    auto HilbertSub = Kokkos::subview( HilbertArray, space.range( 0 ), space.range( 1 ), space.range( 2 ), space.range( 3 ) );
    auto RegularSub = Kokkos::subview( RegularArray, space.range( 0 ), space.range( 1 ), space.range( 2 ), space.range( 3 )  );

    std::cout << "Hilbert SubView Var 1\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < 2; j++ ) {
            std::cout << HilbertSub( i, j, 0, 0 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Hilbert SubView Var 2\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < 2; j++ ) {
            std::cout << HilbertSub( i, j, 0, 1 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Regular SubView Var 1\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < 2; j++ ) {
            std::cout << RegularSub( i, j, 0, 0 ) << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Regular SubView Var 2\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < 2; j++ ) {
            std::cout << RegularSub( i, j, 0, 1 ) << " ";
        }
        std::cout << "\n";
    }

    Kokkos::View<double****> RegularSmall( "Regular", dim1, 2, dim3, dim4 );
    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < 2; j++ ) {
            for ( int k = 0; k < dim3; k++ ) {
                for ( int l = 0; l < dim4; l++ ) {
                    RegularSmall( i, j, k, l ) = 7012;
                }
            }
        }
    }

    Kokkos::deep_copy( HilbertSub,  RegularSmall );

    std::cout << "Hilbert Var 1\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << HilbertArray( i, j, 0, 0 ) << "\t";
        }
        std::cout << "\n";
    }

    std::cout << "Hilbert Var 2\n";

    for ( int i = 0; i < dim1; i++ ) {
        for ( int j = 0; j < dim2; j++ ) {
            std::cout << HilbertArray( i, j, 0, 1 ) << "\t";
        }
        std::cout << "\n";
    }

    }

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
};