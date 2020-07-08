#include <Kokkos_Core.hpp>
#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <mpi.h>

int main( int argc, char* argv[] ) {
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );
    {
    std::cout << "Testing Hilbert Layout\n"; 

    std::cout << "HilbertArray\n";
    Kokkos::View<double****, Kokkos::LayoutHilbertRight, Kokkos::HostSpace> HilbertArray( "Hilbert", 6, 4, 1, 1 );
    Kokkos::View<double****> RegularArray( "Regular", 6, 4, 1, 1 );

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            HilbertArray( i, j, 0, 0 ) = i + 6 * j;
            RegularArray( i, j, 0, 0 ) = i + 6 * j;
        }
    }

    std::cout << "Hilbert Array\n";

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << i << " " << j << " " << HilbertArray( i, j, 0, 0 ) << "\n";
        }
    }

    std::cout << "Regular Array\n";

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << i << " " << j << " " << RegularArray( i, j, 0, 0 ) << "\n";
        }
    }

    Cajita::IndexSpace<4> space;
    space = Cajita::IndexSpace<4>( { 0, 0, 0, 0 }, { 3, 3, 1, 1 } );

    auto HilbertSub = Kokkos::subview( HilbertArray, space.range( 0 ), space.range( 1 ), space.range( 2 ), space.range( 3 ) );
    auto RegularSub = Kokkos::subview( RegularArray, space.range( 0 ), space.range( 1 ), space.range( 2 ), space.range( 3 )  );

    std::cout << "Hilbert SubView\n";

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            std::cout << i << " " << j << " " << HilbertSub( i, j, 0, 0 ) << "\n";
        }
    }

    std::cout << "Regular Array\n";

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            std::cout << i << " " << j << " " << RegularSub( i, j, 0, 0 ) << "\n";
        }
    }

    }

    // Cajita::Hilbert<double, Kokkos::Device<Kokkos::Serial, Kokkos::HostSpace>, Cabana::DefaultAccessMemory, 2> HilbertTest();

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
};