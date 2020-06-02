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
}
template<class MemorySpace, class state_t>
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

        template<class InitFunc, class ExecutionSpace>
        ProblemManager( const std::shared_ptr<Mesh<MemorySpace>>& mesh, const InitFunc& create_functor, const ExecutionSpace& exec_space ) 
        : _mesh ( mesh )
        {
            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() );
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            _velocityA = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _heightA = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );
            _velocityB = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _heightB = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _cell_vector_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_vector_layout, Cajita::HaloPattern() );
            _cell_scalar_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_scalar_layout, Cajita::HaloPattern() );

            initialize( create_functor, exec_space );

        };

        template<class InitFunctor, class ExecutionSpace>
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
                int i_own = i - owned_cells.min( Cajita::Dim::I );
                int j_own = j - owned_cells.min( Cajita::Dim::J );
                int k_own = k - owned_cells.min( Cajita::Dim::K );

                /*
                printf("Rank: %d\tExtent: %d, %d, %d\n", _mesh->rank(), owned_cells.extent(0), owned_cells.extent(1), owned_cells.extent(2));
                printf( "Rank: %d\ti: %d\tj: %d\tk: %d\tiown: %d\tjown: %d\tkown: %d\n", _mesh->rank(), i, j, k, i_own, j_own, k_own);
                */
                
                int coords[3] = { i, j, k };
                state_t x[3];
                local_mesh.coordinates( Cajita::Cell(), coords, x );

                state_t velocity[2];
                state_t height;

                create_functor(x, velocity, height);

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

        const std::shared_ptr<Mesh<MemorySpace>>& mesh() const {
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

        void scatter( Location::Cell, Field::Velocity, int t ) const {
            if ( t == 0 ) _cell_vector_halo->scatter( *_velocityA );
            else _cell_vector_halo->scatter( *_velocityB );
        };

        void scatter( Location::Cell, Field::Height, int t ) const {
            if ( t == 0 ) _cell_scalar_halo->scatter( *_heightA );
            else _cell_scalar_halo->scatter( *_heightB );
        };

    private:
#if 0
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
#endif
        std::shared_ptr<Mesh<MemorySpace>> _mesh;
        std::shared_ptr<cell_array> _velocityA;
        std::shared_ptr<cell_array> _heightA;
        std::shared_ptr<cell_array> _velocityB;
        std::shared_ptr<cell_array> _heightB;
        std::shared_ptr<halo> _cell_vector_halo;
        std::shared_ptr<halo> _cell_scalar_halo;
};

}

#endif
