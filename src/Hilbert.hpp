#ifndef HILBERT_HPP
#define HILBERT_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif 

#include <Kokkos_Core.hpp>

namespace Kokkos
{

    struct LayoutHilbert {
        typedef LayoutHilbert array_layout;

        size_t dimension[ARRAY_LAYOUT_MAX_RANK];

        enum { is_extent_constructible = true };

        LayoutHilbert( LayoutHilbert const & ) = default;
        LayoutHilbert( LayoutHilbert && ) = default;
        LayoutHilbert& operator=( LayoutHilbert const & ) = default;
        LayoutHilbert& operator=( LayoutHilbert && ) = default;

        KOKKOS_INLINE_FUNCTION
        explicit constexpr LayoutHilbert( size_t N0 = 0, size_t N1 = 0, size_t N2 = 0,
                                            size_t N3 = 0, size_t N4 = 0, size_t N5 = 0,
                                            size_t N6 = 0, size_t N7 = 0 )
        : dimension{ N0, N1, N2, N3, N4, N5, N6, N7 }
        {}

    };

    namespace Impl
    {
        struct HilbertMap2D {
            int hilbert_step;
            std::vector<int> map;

            HilbertMap2D() {};

            HilbertMap2D( long unsigned int width, long unsigned int height ) {
                std::cout << "HilbertMap2D: " << width << " " << height << "\n";
                if ( width > 0 && height > 0 ) {
                    hilbert_step = 0;
                    map.resize( width * height );

                    if ( width >= height ) {
                        gilbert2d( 0, 0, width, 0, 0, height, width, height );
                    }
                    else {
                        gilbert2d( 0, 0, 0, height, width, 0, width, height );
                    }
                }
            };

            void gilbert2d( int x, int y, int ax, int ay, int bx, int by, int nx, int ny ) {
                int w = std::abs( ax + ay );
                int h = std::abs( bx + by );

                int dax = ( ( ax > 0 ) - ( ax < 0 ) );
                int day = ( ( ay > 0 ) - ( ay < 0 ) );
                int dbx = ( ( bx > 0 ) - ( bx < 0 ) );
                int dby = ( ( by > 0 ) - ( by < 0 ) );

                if ( h == 1 ) {
                    for ( int i = 0; i < w; i++ ) {
                        if ( DEBUG ) std::cout << "( " << x << ", " << y << " )\t" << x + nx * y << "\t" << hilbert_step << "\n";
                        map[ x + nx * y ] = hilbert_step;
                        hilbert_step ++;
                        x += dax;
                        y += day;
                    }
                    return;
                }

                if ( w == 1 ) {
                    for ( int i = 0; i < h; i++ ) {
                        if ( DEBUG ) std::cout << "( " << x << ", " << y << " )\t" << x + nx * y << "\t" << hilbert_step << "\n";
                        map[ x + nx * y ] = hilbert_step;
                        hilbert_step ++;
                        x += dbx;
                        y += dby;
                    }
                    return;
                }

                int ax2 = ax / 2;
                int ay2 = ay / 2;
                int bx2 = bx / 2;
                int by2 = by / 2;

                int w2 = std::abs( ax2 + ay2 );
                int h2 = std::abs( bx2 + by2 );

                if ( 2 * w > 3 * h ) {
                    if ( ( w2 % 2 ) && ( w > 2 ) ) {
                        ax2 += dax;
                        ay2 += day;
                    }

                    gilbert2d( x, y, ax2, ay2, bx2, by2, nx, ny );
                    gilbert2d( x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by, nx, ny );
                }
                else {
                    if ( ( h2 % 2 ) && ( h > 2 ) ) {
                        bx2 += dbx;
                        by2 += dby;
                    }

                    gilbert2d( x, y, bx2, by2, ax2, ay2, nx, ny );
                    gilbert2d( x + bx2, y + by2, ax, ay, bx - bx2, by - by2, nx, ny );
                    gilbert2d( x + ( ax - dax ) + ( bx2 - dbx ), y + ( ay - day ) + ( by2 - dby ), -bx2, -by2, - ( ax - ax2 ), - ( ay - ay2 ), nx, ny );
                }
            };
        };

        template <class Dimension>
        struct ViewOffset<Dimension, Kokkos::LayoutHilbert, void> {

            using is_mapping_plugin = std::true_type;
            using is_regular        = std::true_type;

            typedef size_t size_type;
            typedef Dimension dimension_type;
            typedef Kokkos::LayoutHilbert array_layout;

            dimension_type m_dim;

            HilbertMap2D hilbert_map{ m_dim.N0, m_dim.N1 };

            // rank 1
            template <typename I0>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0 ) const {
                return i0;
            };

            // rank 2
            template <typename I0, typename I1>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1 ) const {
                std::cout << i0 + m_dim.N0 * i1 << "\t";

                int hilbert = hilbert_map.map[i0 + m_dim.N0 * i1];

                std::cout << "Hilbert Index: " << hilbert << "\t";

                return hilbert;
            };

            // rank 3
            template <typename I0, typename I1, typename I2>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2 ) const {
                std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * i2 ) << "\t";
                
                int hilbert = hilbert_map.map[i0 + m_dim.N0 * i1];

                std::cout << "Hilbert Index: " << m_dim.N0 * m_dim.N1 * i2 + hilbert << "\t";

                return m_dim.N0 * m_dim.N1 * i2 + hilbert;
            };

            // rank 4
            template <typename I0, typename I1, typename I2, typename I3>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3 ) const {
                std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) ) << "\t";

                int hilbert = hilbert_map.map[i0 + m_dim.N0 * i1];

                std::cout << "Hilbert Index: " << ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * i3 ) + hilbert << "\t";

                return ( m_dim.N0 * m_dim.N1 ) * ( i2 + m_dim.N2 * i3 ) + hilbert;
            };

            // rank 5
            template <typename I0, typename I1, typename I2, typename I3, typename I4>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4 ) const {
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * i4 ) ) );
            };

            // rank 6
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5 ) const {
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * i5 ) ) ) );
            };

            // rank 7
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5, I6 const& i6 ) const {
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * i6 ) ) ) ) );
            };

            // rank 8
            template <typename I0, typename I1, typename I2, typename I3, typename I4, typename I5, typename I6, typename I7>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3, I4 const& i4, I5 const& i5, I6 const& i6, I7 const& i7 ) const {
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * ( i3 + m_dim.N3 * ( i4 + m_dim.N4 * ( i5 + m_dim.N5 * ( i6 + m_dim.N6 * i7 ) ) ) ) ) );
            };

            KOKKOS_INLINE_FUNCTION
            constexpr array_layout layout() const {
                return array_layout( m_dim.N0, m_dim.N1, m_dim.N2, m_dim.N3, m_dim.N4, m_dim.N5, m_dim.N6, m_dim.N7 );
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_0() const {
                return m_dim.N0;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_1() const {
                return m_dim.N1;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_2() const {
                return m_dim.N2;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_3() const {
                return m_dim.N3;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_4() const {
                return m_dim.N4;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_5() const {
                return m_dim.N5;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_6() const {
                return m_dim.N6;
            };

            KOKKOS_INLINE_FUNCTION constexpr size_type dimension_7() const {
                return m_dim.N7;
            };

            /* Cardinality of the domain index space */
            KOKKOS_INLINE_FUNCTION
            constexpr size_type size() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
                    m_dim.N6 * m_dim.N7;
            };

            /* Span of the range space */
            KOKKOS_INLINE_FUNCTION
            constexpr size_type span() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 *
                    m_dim.N6 * m_dim.N7;
            };

            KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
                return true;
            };

            /* Strides of dimensions */
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_0() const { 
                return 1; 
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_1() const {
                return m_dim.N0;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_2() const {
                return m_dim.N0 * m_dim.N1;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_3() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_4() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_5() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_6() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5;
            };

            KOKKOS_INLINE_FUNCTION 
            constexpr size_type stride_7() const {
                return m_dim.N0 * m_dim.N1 * m_dim.N2 * m_dim.N3 * m_dim.N4 * m_dim.N5 * m_dim.N6;
            };

            // Stride with [ rank ] value is the total length
            template <typename iType>
            KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
                s[0] = 1;
                if (0 < dimension_type::rank) {
                    s[1] = m_dim.N0;
                }
                if ( 1 < dimension_type::rank ) {
                    s[2] = s[1] * m_dim.N1;
                }
                if ( 2 < dimension_type::rank ) {
                    s[3] = s[2] * m_dim.N2;
                }
                if ( 3 < dimension_type::rank ) {
                    s[4] = s[3] * m_dim.N3;
                }
                if ( 4 < dimension_type::rank ) {
                    s[5] = s[4] * m_dim.N4;
                }
                if ( 5 < dimension_type::rank ) {
                    s[6] = s[5] * m_dim.N5;
                }
                if ( 6 < dimension_type::rank ) {
                    s[7] = s[6] * m_dim.N6;
                }
                if ( 7 < dimension_type::rank ) {
                    s[8] = s[7] * m_dim.N7;
                }
            };

            ViewOffset()                  = default;
            ViewOffset( const ViewOffset& ) = default;
            ViewOffset& operator=( const ViewOffset& ) = default;

            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( std::integral_constant<unsigned, 0> const&, Kokkos::LayoutHilbert const& rhs)
            : m_dim( rhs.dimension[0], rhs.dimension[1], rhs.dimension[2], rhs.dimension[3], rhs.dimension[4], rhs.dimension[5], rhs.dimension[6], rhs.dimension[7] ) {};

            template <class DimRHS>
            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( const ViewOffset<DimRHS, Kokkos::LayoutHilbert, void>& rhs )
            : m_dim( rhs.m_dim.N0, rhs.m_dim.N1, rhs.m_dim.N2, rhs.m_dim.N3, rhs.m_dim.N4, rhs.m_dim.N5, rhs.m_dim.N6, rhs.m_dim.N7 ) {
                static_assert( int( DimRHS::rank ) == int( dimension_type::rank ), "ViewOffset assignment requires equal rank" );
                // Also requires equal static dimensions ...
            };

            template <class DimRHS>
            KOKKOS_INLINE_FUNCTION 
            ViewOffset( const ViewOffset<DimRHS, Kokkos::LayoutStride, void>& rhs )
            : m_dim( rhs.m_dim.N0, 0, 0, 0, 0, 0, 0, 0 ) {
                if ( rhs.m_stride.S0 != 1 ) {
                Kokkos::abort( "Kokkos::Impl::ViewOffset assignment of LayoutHilbert from LayoutStride requires stride == 1" );
                }
            };

            //----------------------------------------
            // Subview construction

            template <class DimRHS>
            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( const ViewOffset<DimRHS, Kokkos::LayoutHilbert, void>&, const SubviewExtents<DimRHS::rank, dimension_type::rank>& sub )
            : m_dim( sub.range_extent(0), 0, 0, 0, 0, 0, 0, 0 ) {
                static_assert( ( 0 == dimension_type::rank_dynamic ) || ( 1 == dimension_type::rank && 1 == dimension_type::rank_dynamic && 1 <= DimRHS::rank ), 
                "ViewOffset subview construction requires compatible rank" );
            };
        };
    }
}

namespace ExaCLAMR
{
    namespace Impl
    {

    }
}

#endif