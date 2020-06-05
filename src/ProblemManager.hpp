#ifndef EXACLAMR_PROBLEMMANAGER_HPP
#define EXACLAMR_PROBLEMMANAGER_HPP

#include <Mesh.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include <memory>

namespace ExaCLAMR
{

namespace Location
{
struct Cell {};
struct Face {};
struct Node {};
}

namespace Field
{
struct Velocity {};
struct Height {};

struct HxFluxPlus {};
struct HxFluxMinus {};
struct UxFluxPlus {};
struct UxFluxMinus {};

struct HyFluxPlus {};
struct HyFluxMinus {};
struct UyFluxPlus {};
struct UyFluxMinus {};

struct HxWPlus {};
struct HxWMinus {};
struct HyWPlus {};
struct HyWMinus {};

struct UWPlus {};
struct UWMinus {};
}

template<class MemorySpace, class ExecutionSpace, class state_t>
class ProblemManager
{
#if 0
    using cell_members = Cabana::MemberTypes<state_t[2], state_t>;
    using cell_list = Cabana::AoSoA<cell_members, MemorySpace>;
    using cell_type = typename cell_list::tuple_type;
#endif 

    using cell_array = Cajita::Array<state_t, Cajita::Cell, Cajita::UniformMesh<state_t>, MemorySpace>;
    using halo = Cajita::Halo<state_t, MemorySpace>;

    public:

        template<class InitFunc>
        ProblemManager( const std::shared_ptr<Mesh<MemorySpace, ExecutionSpace>>& mesh, const InitFunc& create_functor, const ExecutionSpace& exec_space ) 
        : _mesh ( mesh )
        {
            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() );
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            _velocityA = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _heightA = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _velocityB = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _heightB = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );


            // Flux Arrays
            _Hxfluxplus = Cajita::createArray<state_t, MemorySpace>( "HxFluxPlus", cell_scalar_layout );
            _Hxfluxminus = Cajita::createArray<state_t, MemorySpace>( "HxFluxMinus", cell_scalar_layout );
            _Uxfluxplus = Cajita::createArray<state_t, MemorySpace>( "UxFluxPlus", cell_vector_layout );
            _Uxfluxminus = Cajita::createArray<state_t, MemorySpace>( "UxFluxMinus", cell_vector_layout );

            _Hyfluxplus = Cajita::createArray<state_t, MemorySpace>( "HyFluxPlus", cell_scalar_layout );
            _Hyfluxminus = Cajita::createArray<state_t, MemorySpace>( "HyFluxMinus", cell_scalar_layout );
            _Uyfluxplus = Cajita::createArray<state_t, MemorySpace>( "UyFluxPlus", cell_vector_layout );
            _Uyfluxminus = Cajita::createArray<state_t, MemorySpace>( "UyFluxMinus", cell_vector_layout );

            // Flux Corrector Arrays
            _Hxwplus = Cajita::createArray<state_t, MemorySpace>( "HxWPlus", cell_scalar_layout );
            _Hxwminus = Cajita::createArray<state_t, MemorySpace>( "HxWMinus", cell_scalar_layout );
            _Hywplus = Cajita::createArray<state_t, MemorySpace>( "HyWPlus", cell_scalar_layout );
            _Hywminus = Cajita::createArray<state_t, MemorySpace>( "HyWMinus", cell_scalar_layout );

            _Uwplus = Cajita::createArray<state_t, MemorySpace>( "UWPlus", cell_vector_layout );
            _Uwminus = Cajita::createArray<state_t, MemorySpace>( "UWMinus", cell_vector_layout );

            auto HaloPattern = Cajita::HaloPattern();
            std::vector<std::array<int, 3>> neighbors;

            // Setting up Stencil ( Left, Right, Top, Bottom )
            for ( int i = -1; i < 2; i++ ) {
                for (int j = -1; j < 2; j++) {
                    if ( ( i == 0 || j == 0 ) && !( i == 0 && j == 0 ) ) {
                        neighbors.push_back( { i, j, 0 } );
                    }
                }
            }

            HaloPattern.setNeighbors( neighbors );

            _cell_vector_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_vector_layout, HaloPattern );
            _cell_scalar_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_scalar_layout, HaloPattern );

            initialize( create_functor, exec_space );

        };

        template<class InitFunctor>
        void initialize( const InitFunctor& create_functor, const ExecutionSpace& exec_space ) {
            if ( _mesh->rank() == 0 ) printf( "Initializing Cell Fields\n" );

            using device_type = typename cell_array::device_type;

            auto local_grid = *( _mesh->localGrid() );
            auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

            /*
            printf( "Rank: %d\tLow Corner: %.4f, %.4f, %.4f\tHigh Corner: %.4f, %.4f, %.4f\n", _mesh->rank(), \
            local_mesh.lowCorner( Cajita::Own() , 0 ),  local_mesh.lowCorner( Cajita::Own() , 1 ), local_mesh.lowCorner( Cajita::Own() , 2 ), \
            local_mesh.highCorner( Cajita::Own() , 0 ), local_mesh.highCorner( Cajita::Own() , 1 ), local_mesh.highCorner( Cajita::Own() , 2 ));
            */

            auto ghost_cells = local_grid.indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );
            auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

	        auto uA = get(Location::Cell(), Field::Velocity(), 0 );
	        auto hA = get(Location::Cell(), Field::Height(), 0 );
            auto uB = get(Location::Cell(), Field::Velocity(), 1 );
	        auto hB = get(Location::Cell(), Field::Height(), 1 );

            Kokkos::parallel_for( "Initializing", Cajita::createExecutionPolicy( ghost_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                /*
                int i_own = i - owned_cells.min( Cajita::Dim::I );
                int j_own = j - owned_cells.min( Cajita::Dim::J );
                int k_own = k - owned_cells.min( Cajita::Dim::K );
                */

                /*
                printf("Rank: %d\tExtent: %d, %d, %d\n", _mesh->rank(), owned_cells.extent(0), owned_cells.extent(1), owned_cells.extent(2));
                printf( "Rank: %d\ti: %d\tj: %d\tk: %d\tiown: %d\tjown: %d\tkown: %d\n", _mesh->rank(), i, j, k, i_own, j_own, k_own);
                */
                
                std::array<double, 6> boundingBox = _mesh->globalBoundingBox();
                
                state_t center[3] = { (boundingBox[3] - boundingBox[0]) / 2, ( boundingBox[4] - boundingBox[1] ) / 2, ( boundingBox[5] - boundingBox[2] ) / 2 };
                // printf( "Center x: %.4f\t Center y: %.4f\tCenter z: %.4f\n", center[0], center[1], center[2] );

                int coords[3] = { i, j, k };
                state_t x[3];

                local_mesh.coordinates( Cajita::Cell(), coords, x );

                state_t r_dist = sqrt( pow( x[0] - center[0], 2 ) + pow( x[1] - center[1], 2 ) + pow( x[2] - center[2], 2 ) );

                // printf( "x: %.4f\ty: %.4f\tz: %.4f\trDist: %.4f\n", x[0], x[1], x[2], r_dist );

                state_t velocity[2];
                state_t height;

                create_functor(r_dist, velocity, height);

                /*
                printf( "Rank: %d\ti: %d\tj: %d\tk: %d\t x: %.4f\ty: %.4f\tz: %.4f\tvx: %.4f\tvy: %.4f\th: %.4f\n", \
                _mesh->rank(), i, j, k, x[0], x[1], x[2], velocity[0], velocity[1], height );
                */

		        uA( i, j, k, 0 ) = velocity[0];
		        uA( i, j, k, 1 ) = velocity[1];
                hA( i, j, k, 0 ) = height;

                uB( i, j, k, 0 ) = velocity[0];
		        uB( i, j, k, 1 ) = velocity[1];
                hB( i, j, k, 0 ) = height;

            } );

            /*
            if ( _mesh->rank() == 0 ) {
                for ( int i = 0; i < owned_cells.extent( 0 ); i++ ) {
                    for ( int j = 0; j < owned_cells.extent( 1 ); j++ ) {
                        for ( int k = 0; k < owned_cells.extent( 2 ); k++ ) {
                            printf( "%.4f\t", hA( i, j, k, 0 ) );
                        }
                    }
                    printf("\n");
                }
            }
            */
            
            

        };

        const std::shared_ptr<Mesh<MemorySpace, ExecutionSpace>>& mesh() const {
            return _mesh;
        };

        typename cell_array::view_type get( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) return _velocityA->view();
            else return _velocityB->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::Height, int t ) const {
            if (t == 0 ) return _heightA->view();
            else return _heightB->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::HxFluxPlus ) const {
            return _Hxfluxplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxFluxMinus ) const {
            return _Hxfluxminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UxFluxPlus ) const {
            return _Uxfluxplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UxFluxMinus ) const {
            return _Uxfluxminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyFluxPlus ) const {
            return _Hyfluxplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyFluxMinus ) const {
            return _Hyfluxminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UyFluxPlus ) const {
            return _Uyfluxplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UyFluxMinus ) const {
            return _Uyfluxminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxWPlus ) const {
            return _Hxwplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HxWMinus ) const {
            return _Hxwminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyWPlus ) const {
            return _Hywplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::HyWMinus ) const {
            return _Hywminus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UWPlus ) const {
            return _Uwplus->view();
        }

        typename cell_array::view_type get( Location::Cell, Field::UWMinus ) const {
            return _Uwminus->view();
        }

        void scatter( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) _cell_vector_halo->scatter( *_velocityA );
            else _cell_vector_halo->scatter( *_velocityB );
        };

        void scatter( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->scatter( *_heightA );
            else _cell_scalar_halo->scatter( *_heightB );
        };

        void gather( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) _cell_vector_halo->gather( *_velocityA );
            else _cell_vector_halo->gather( *_velocityB );
        }

        void gather( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->gather( *_heightA );
            else _cell_scalar_halo->gather( *_heightB );
        }

    private:
#if 0
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
#endif
        std::shared_ptr<Mesh<MemorySpace, ExecutionSpace>> _mesh;
        std::shared_ptr<cell_array> _velocityA;
        std::shared_ptr<cell_array> _heightA;
        std::shared_ptr<cell_array> _velocityB;
        std::shared_ptr<cell_array> _heightB;

        std::shared_ptr<cell_array> _Hxfluxplus;
        std::shared_ptr<cell_array> _Hxfluxminus;
        std::shared_ptr<cell_array> _Uxfluxplus;
        std::shared_ptr<cell_array> _Uxfluxminus;

        std::shared_ptr<cell_array> _Hyfluxplus;
        std::shared_ptr<cell_array> _Hyfluxminus;
        std::shared_ptr<cell_array> _Uyfluxplus;
        std::shared_ptr<cell_array> _Uyfluxminus;

        std::shared_ptr<cell_array> _Hxwplus;
        std::shared_ptr<cell_array> _Hxwminus;
        std::shared_ptr<cell_array> _Hywplus;
        std::shared_ptr<cell_array> _Hywminus;

        std::shared_ptr<cell_array> _Uwplus;
        std::shared_ptr<cell_array> _Uwminus;

        std::shared_ptr<halo> _cell_vector_halo;
        std::shared_ptr<halo> _cell_scalar_halo;
};

}

#endif
