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

            _velocity = Cajita::createArray<state_t, MemorySpace>( "velocity", cell_vector_layout );
            _height = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _cell_vector_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_vector_layout, Cajita::HaloPattern() );
            _cell_scalar_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_scalar_layout, Cajita::HaloPattern() );

            initialize( create_functor, exec_space );

        };

        template<class InitFunctor, class ExecutionSpace>
        void initialize( const InitFunctor& create_functor, const ExecutionSpace& exec_space ) {
            printf( "Initializing\n" );

            using device_type = typename cell_array::device_type;

            auto domain = _mesh->maxDomainGlobalCellIndex();
            printf( "Mesh Info: Domain Size: %d, %d, %d\n",  domain[0], domain[1], domain[2] );

            auto local_grid = *( _mesh->localGrid() );
            auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

            auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

	        auto u_i = get(Location::Cell(), Field::Velocity());
	        auto h_i = get(Location::Cell(), Field::Height());

            Kokkos::parallel_for( "Initializing", Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                int cid = i + owned_cells.extent( 1 ) * ( j + owned_cells.extent( 2 ) * k );

                /*
                printf("Extent: %d, %d, %d\n", owned_cells.extent(0), owned_cells.extent(1), owned_cells.extent(2));
                printf( "i: %d\tj: %d\tk: %d\tpid: %d\n", i, j, k, cid );
                */
               
                int coords[3] = { i, j, k };
                state_t x[3];
                local_mesh.coordinates( Cajita::Cell(), coords, x );

                state_t velocity[2];
                state_t height;

                create_functor(x, velocity, height);

		        u_i(i, j, k, 0) = velocity[0];
		        u_i(i, j, k, 1) = velocity[1];
                h_i(i, j, k, 0) = height;

            });

        };

        const std::shared_ptr<Mesh<MemorySpace>>& mesh() const {
            return _mesh;
        };

        typename cell_array::view_type get( Location::Cell, Field::Velocity ) const {
            return _velocity->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::Height ) const {
            return _height->view();
        }

        void scatter( Location::Cell, Field::Velocity ) const {
            _cell_vector_halo->scatter( *_velocity );
        }

        void scatter( Location::Cell, Field::Height ) const {
            _cell_scalar_halo->scatter( *_height );
        }

    private:
#if 0
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
#endif
        std::shared_ptr<Mesh<MemorySpace>> _mesh;
        std::shared_ptr<cell_array> _velocity;
        std::shared_ptr<cell_array> _height;
        std::shared_ptr<halo> _cell_vector_halo;
        std::shared_ptr<halo> _cell_scalar_halo;
};

}

#endif
