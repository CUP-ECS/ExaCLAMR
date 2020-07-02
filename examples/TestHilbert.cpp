#include <Kokkos_Core.hpp>
#include <Hilbert.hpp>

int main( int argc, char* argv[] ) {
    Kokkos::initialize( argc, argv );
    {
    std::cout << "Testing Hilbert Layout\n"; 

    std::cout << "Regular Array\n";
    Kokkos::View<double**> RegularArray( "Regular", 4, 4 );
    
    for ( int i = 0; i < 4; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << "i: " << i << "\tj: " << j << "\tFlat Index: " << i + 4 * j << "\n";
        }
    }

    std::cout << "HilbertArray\n";
    Kokkos::View<double**, Kokkos::LayoutHilbert, Kokkos::HostSpace> HilbertArray( "Hilbert", 4, 4 );

    for ( int i = 0; i < 4; i++ ) {
        for ( int j = 0; j < 4; j++ ) {
            std::cout << "i: " << i << "\tj: " << j << "\tFlat Index: " << HilbertArray( i, j ) << "\n";
        }
    }
    

    }
    Kokkos::finalize();

    return 0;
};