#ifndef HILBERT_HPP
#define HILBERT_HPP

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
        template <class Dimension>
        struct ViewOffset<Dimension, Kokkos::LayoutHilbert, void> {

            using is_mapping_plugin = std::true_type;
            using is_regular        = std::true_type;

            typedef size_t size_type;
            typedef Dimension dimension_type;
            typedef Kokkos::LayoutHilbert array_layout;

            dimension_type m_dim;

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

                int rotx, roty, s, temp, hilbert=0;
                I0 x = i0;
                I1 y = i1;

                size_type nx = dimension_0();
                size_type ny = dimension_1();

                for ( s = nx / 2; s > 0; s /= 2 ) {
                    rotx = ( x & s ) > 0;
                    roty = ( y & s ) > 0;
                    hilbert += s * s * ( ( 3 * rotx ) ^ roty );

                    if ( roty == 0 ) {
                        if ( rotx == 1 ) {
                            x = nx - 1 - y;
                            y = ny - 1 - x;
                        }
                    }

                    temp = x;
                    x = y;
                    y = temp;
                }

                std::cout << "Hilbert Index: " << hilbert << "\t";

                return i0 + m_dim.N0 * i1;
            };

            // rank 3
            template <typename I0, typename I1, typename I2>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2 ) const {
                std::cout << i0 + m_dim.N0 * ( i1 + m_dim.N1 * i2 ) << "\t";
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * i2 );
            };

            // rank 4
            template <typename I0, typename I1, typename I2, typename I3>
            KOKKOS_INLINE_FUNCTION 
            constexpr size_type operator()( I0 const& i0, I1 const& i1, I2 const& i2, I3 const& i3 ) const {
                return i0 + m_dim.N0 * ( i1 + m_dim.N1 * ( i2 + m_dim.N2 * i3 ) );
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
                return array_layout(m_dim.N0, m_dim.N1, m_dim.N2, m_dim.N3, m_dim.N4, m_dim.N5, m_dim.N6, m_dim.N7);
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

            template <unsigned TrivialScalarSize>
            KOKKOS_INLINE_FUNCTION 
            constexpr ViewOffset( std::integral_constant<unsigned, TrivialScalarSize> const&, Kokkos::LayoutHilbert const& arg_layout)
            : m_dim( arg_layout.dimension[0], 0, 0, 0, 0, 0, 0, 0 ) {};

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