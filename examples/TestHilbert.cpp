#include <Kokkos_Core.hpp>
#include <Hilbert.hpp>

int main( int argc, char* argv[] ) {
    Kokkos::initialize( argc, argv );
    {
    std::cout << "Testing Hilbert Layout\n"; 

    std::cout << "HilbertArray\n";
    Kokkos::View<double**, Kokkos::LayoutHilbert, Kokkos::HostSpace> HilbertArray( "Hilbert", 100, 100 );

    for ( int i = 0; i < 18; i++ ) {
        for ( int j = 0; j < 18; j++ ) {
            std::cout << "i: " << i << "\tj: " << j << "\tFlat Index: " << HilbertArray( i, j ) << "\n";
        }
    }
    
    }
    Kokkos::finalize();

    return 0;
};