#include <Kokkos_Core.hpp>
#include <Hilbert.hpp>

int sgn( int x ) {
    return ( x > 0 ) - ( x < 0 );
}

void gilbert2d( int x, int y, int ax, int ay, int bx, int by ) {
    int w = abs( ax + ay );
    int h = abs( bx + by );

    int dax = sgn( ax );
    int day = sgn( ay );
    int dbx = sgn( bx );
    int dby = sgn( by );

    if ( h == 1 ) {
        for ( int i = 0; i < w; i++ ) {
            std::cout << "( " << x << ", " << y << " )\n";
            x = x + dax;
            y = y + day;
        }
        return;
    }

    if ( w == 1 ) {
        for ( int i = 0; i < h; i++ ) {
            std::cout << "( " << x << ", " << y << " )\n";
            x = x + dbx;
            y = y + dby;
        }
        return;
    }

    int ax2 = ax / 2;
    int ay2 = ay / 2;
    int bx2 = bx / 2;
    int by2 = by / 2;

    int w2 = abs( ax2 + ay2 );
    int h2 = abs( bx2 + by2 );

    if ( 2 * w > 3 * h ) {
        if ( ( w2 % 2 ) && ( w > 2 ) ) {
            ax2 = ax2 + dax;
            ay2 = ay2 + day;
        }
        gilbert2d( x, y, ax2, ay2, bx2, by2 );
        gilbert2d( x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by );
    }
    else {
        if ( ( h2 % 2 ) && ( h > 2 ) ) {
            bx2 = bx2 + dbx;
            by2 = by2 + dby;
        }

        gilbert2d( x, y, bx2, by2, ax2, ay2 );
        gilbert2d( x + bx2, y + by2, ax, ay, bx - bx2, by - by2 );
        gilbert2d( x + ( ax - dax ) + ( bx2 - dbx ), y + ( ay - day ) + ( by2 - dby ), -bx2, -by2, - ( ax - ax2 ), - ( ay - ay2 ) );
    }
}

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

    int width = 6;
    int height = 6;

    if ( width >= height ) {
        gilbert2d( 0, 0, width, 0, 0, height );
    }
    else {
        gilbert2d( 0, 0, 0, height, width, 0 );
    }
    

    }
    Kokkos::finalize();

    return 0;
};