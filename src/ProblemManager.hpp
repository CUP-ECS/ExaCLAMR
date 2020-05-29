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
struct Momentum {};
struct Height {};
}
template<class MemorySpace, class state_t>
class ProblemManager
{
    using cell_members = Cabana::MemberTypes<state_t[2], state_t>;
    using cell_list = Cabana::AoSoA<cell_members, MemorySpace>;
    using cell_type = typename cell_list::tuple_type;

    using cell_array = Cajita::Array<state_t, Cajita::Cell, Cajita::UniformMesh<state_t>, MemorySpace>;
    using halo = Cajita::Halo<state_t, MemorySpace>;

    public:

        template<class InitFunc, class ExecutionSpace>
        ProblemManager( const std::shared_ptr<Mesh<MemorySpace>>& mesh, const InitFunc& create_functor, const ExecutionSpace& exec_space ) 
        : _cells ( "cells" ), _mesh ( mesh )
        {
            initialize( create_functor, exec_space );

            auto cell_vector_layout = Cajita::createArrayLayout( _mesh->localGrid(), 2, Cajita::Cell() );
            auto cell_scalar_layout = Cajita::createArrayLayout( _mesh->localGrid(), 1, Cajita::Cell() );

            _momentum = Cajita::createArray<state_t, MemorySpace>( "momentum", cell_vector_layout );
            _height = Cajita::createArray<state_t, MemorySpace>( "height", cell_scalar_layout );

            _cell_vector_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_vector_layout, Cajita::HaloPattern() );
            _cell_scalar_halo = Cajita::createHalo<state_t, MemorySpace>( *cell_scalar_layout, Cajita::HaloPattern() );
        };

        template<class InitFunctor, class ExecutionSpace>
        void initialize( const InitFunctor& create_functor, const ExecutionSpace& exec_space ) {
            printf( "Initializing\n" );

            using device_type = typename cell_list::device_type;
            using cell_type = typename cell_list::tuple_type;

            auto domain = _mesh->maxDomainGlobalNodeIndex();
            printf( "Mesh Info: Domain Size: %d, %d, %d\n",  domain[0], domain[1], domain[2] );

            auto local_grid = *( _mesh->localGrid() );
            auto local_mesh = Cajita::createLocalMesh<device_type>( local_grid );

            auto owned_cells = local_grid.indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

            Kokkos::parallel_for( "Initializing", Cajita::createExecutionPolicy( owned_cells, exec_space ), KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                int cid = i + owned_cells.extent( 1 ) * ( j + owned_cells.extent( 2 ) * k );

                /*
                printf("Extent: %d, %d, %d\n", owned_cells.extent(0), owned_cells.extent(1), owned_cells.extent(2));
                printf( "i: %d\tj: %d\tk: %d\tpid: %d\n", i, j, k, cid );
                */

                double momentum[2];
                double height;

                cell_type cell;

                create_functor( momentum, height, cell );

                _cells.setTuple( cid, cell );

            });

        };

        const std::shared_ptr<Mesh<MemorySpace>>& mesh() const {
            return _mesh;
        };

        typename cell_array::view_type get( Location::Cell, Field::Momentum ) const {
            return _momentum->view();
        };

        typename cell_array::view_type get( Location::Cell, Field::Height ) const {
            return _height->view();
        }

        void scatter( Location::Cell, Field::Momentum ) const {
            _cell_vector_halo->scatter( *_momentum );
        }

        void scatter( Location::Cell, Field::Height ) const {
            _cell_scalar_halo->scatter( *_height );
        }

    private:
        Cabana::AoSoA<cell_members, MemorySpace> _cells;
        std::shared_ptr<Mesh<MemorySpace>> _mesh;
        std::shared_ptr<cell_array> _momentum;
        std::shared_ptr<cell_array> _height;
        std::shared_ptr<halo> _cell_vector_halo;
        std::shared_ptr<halo> _cell_scalar_halo;
};

}

#endif