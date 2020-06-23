/**
 * @file
 * @author Patrick Bridges <pbridges@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Mesh class that stores rank, bounding box, domain index spaces and Cajita local grid
 */

#ifndef EXACLAMR_MESH_HPP
#define EXACLAMR_MESH_HPP

#ifndef DEBUG
    #define DEBUG 0 
#endif 

// Include Statements
#include <Cajita.hpp>
#include <Kokkos_Core.hpp>

#include<mpi.h>

#include <memory>


namespace ExaCLAMR
{
/**
 * The Mesh Class
 * @class Mesh
 * @brief Mesh class to handle and wrap the Cajita Regular Grid class and 
 * to track the domain index spaces and track which cells are boundary cells.
 **/
template <class MemorySpace, class ExecutionSpace, typename state_t>
class Mesh
{
    public:
        /**
         * Constructor
         * Creates a new mesh and calculates the bounding box, number of cells, and the domain index space.
         * Creates a Cajita Local Grid on each rank as a class member
         * 
         * @param cl Command line arguments
         * @param partitioner Cajita MPI partitioner
         * @param comm MPI communicator
         */
        Mesh( const ExaCLAMR::ClArgs<state_t>& cl,
                const Cajita::Partitioner& partitioner,
                MPI_Comm comm ) :
                _global_bounding_box ( cl.global_bounding_box ) {
            // Define device_type for Later Use
            using device_type = typename Kokkos::Device<ExecutionSpace, MemorySpace>;

            MPI_Comm_rank( comm, &_rank );

            // 2-D Mesh - Ignore Z Cells
            std::array<int, 3> num_cell;
            num_cell[0] = cl.global_num_cells[0];   // Set X Num Cells
            num_cell[1] = cl.global_num_cells[1];   // Set Y Num Cells
            num_cell[2] = 1;                        // Set Z Num Cells to 1

            // 2-D Mesh - Z Domain is From 0 to 1
            std::array<state_t, 6> bounding_box;
            bounding_box[0] = cl.global_bounding_box[0];    // Set X Min Coordinate
            bounding_box[1] = cl.global_bounding_box[1];    // Set Y Min Coordinate
            bounding_box[2] = 0;                            // Set Z Min Coordinate to 0
            bounding_box[3] = cl.global_bounding_box[3];    // Set X Max Coordinate
            bounding_box[4] = cl.global_bounding_box[4];    // Set Y Max Coordinate
            bounding_box[5] = 1;                            // Set Z Max Coordinate to 1

            // Calculate Cell Size
            std::array<state_t, 3> cell_size;
            cell_size[0] = ( bounding_box[3] - bounding_box[0] ) / num_cell[0];     // X Cell Size
            cell_size[1] = ( bounding_box[4] - bounding_box[1] ) / num_cell[1];     // Y Cell Size
            cell_size[2] = ( bounding_box[5] - bounding_box[2] ) / num_cell[2];     // Z Cell Size

            // Check if Evenly Divisble by Cell Size
            for ( int dim = 0; dim < 3; ++dim ) {
                state_t extent = ( state_t ) num_cell[dim] * cell_size[dim];
                if ( std::abs( extent - ( bounding_box[dim + 3] - bounding_box[dim] ) ) > state_t( 10.0 ) * std::numeric_limits<state_t>::epsilon() )
                    throw std::logic_error( "Extent not evenly divisible by uniform cell size" );
            }
            
            // DEBUG: Print Num Cells, Bounding Box, and Cell Size ( All 3 Dimensions )
            if ( _rank == 0 && DEBUG ) {
                std::cout << "Num Cells: x: " << num_cell[0] << "\ty: " << num_cell[1] << "\tz: " << num_cell[2] << "\n";
                std::cout << "Bounding Box: xl: " << bounding_box[0] << "\tyl: " << bounding_box[1] << "\tzl: " << bounding_box[2] <<\
                 "\txu: " << bounding_box[3] << "\tyu: " << bounding_box[4] << "\tzu: " << bounding_box[5] << "\n";
                std::cout << "Cell Size: x: " << cell_size[0] << "\ty: " << cell_size[1] << "\tz: " << cell_size[2] << "\n";
            }
            
            // Set Lower and Upper Corner Array
            std::array<state_t, 3> global_low_corner = { bounding_box[0], bounding_box[1], bounding_box[2] };
            std::array<state_t, 3> global_high_corner = { bounding_box[3], bounding_box[4], bounding_box[5] };

            // Adjust Global Corners and Number of Cells for Halo Cells
            for ( int dim = 0; dim < 3; ++dim ) {
                if ( !cl.periodic[dim] && num_cell[dim] != 1 ) {
                    global_low_corner[dim] -= cell_size[dim] * cl.halo_size;
                    global_high_corner[dim] += cell_size[dim] * cl.halo_size;
                    num_cell[dim] += 2 * cl.halo_size;
                }
            }

            // Create Global Mesh and Global Grid - Need to Create Local Grid
            auto global_mesh = Cajita::createUniformGlobalMesh( global_low_corner, global_high_corner, cell_size );
            auto global_grid = Cajita::createGlobalGrid( comm, global_mesh, cl.periodic, partitioner );

            // DEBUG: Print Global Grid Rank, Number of Cells, and Index Offset
            if ( DEBUG ) std::cout << "Global Grid: Rank: " << global_grid->blockId() << \
            "\tNx: " << global_grid->ownedNumCell( 0 ) << "\tNy: " << global_grid->ownedNumCell( 1 ) << "\tNz: " << global_grid->ownedNumCell( 2 ) << \
            "\tOffset x: " << global_grid->globalOffset( 0 ) << "\tOffset y: " << global_grid->globalOffset( 1 ) << "\tOffset z: " << global_grid->globalOffset( 2 ) << "\n";

            // Create Local Grid
            _local_grid = Cajita::createLocalGrid( global_grid, cl.halo_size );

            // Get Local Mesh and Owned Cells for Domain Calculation
            auto local_mesh = Cajita::createLocalMesh<device_type>( *_local_grid );
            auto ghost_cells = _local_grid->indexSpace( Cajita::Ghost(), Cajita::Cell(), Cajita::Local() );

            // Calculate domain index space - assumes regular grid
            for ( int i = 0; i < 3; i ++ ) {
                if ( ghost_cells.max( i ) != 1 ) {
                    _domainMin[i] = ghost_cells.min( i ) + cl.halo_size;
                    _domainMax[i] = ghost_cells.max( i ) - cl.halo_size;
                }
                else {
                    _domainMin[i] = 0;
                    _domainMax[i] = 1;
                }
            }
        };

        /**
         * Returns cell size in given dimension
         * @param dim Dimension of interest
         * @return Cell size of the grid in the given dimension
         **/ 
        state_t cellSize( int dim ) const {
            return _local_grid->globalGrid().globalMesh().cellSize( dim );
        };

        /**
         * Returns the Cajita Local Grid
         * @return The shared pointer to the Cajita Local Grid
         **/
        const std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<state_t>>>& localGrid() const {
            return _local_grid;
        };

        /**
         * Returns the global bounding box array
         * @return The array of length 6 of the global bounding box {xmin, ymin, zmin, xmax, ymax, zmax}
         **/ 
        const std::array<state_t, 6> globalBoundingBox() const {
            return _global_bounding_box;
        };

        /**
         * Returns the rank of the mesh
         * @return The rank of the mesh
         **/
        int rank() const {
            return _rank;
        };

        /**
         * Returns the index space of the domain
         * @return The index space of the domain (excluding ghost and halo cells)
         **/
        const Cajita::IndexSpace<3> domainSpace() const {
            return Cajita::IndexSpace<3>( _domainMin, _domainMax );
        };

        /**
         * Determine whether the cell is on the left boundary
         * @param i Index in x-direction
         * @param j Index in y-direction
         * @param k Index in z-direction
         * @return Returns true if the cell is on the left boundary
         **/
        bool isLeftBoundary( const int i, const int j, const int k ) const {
            return ( j == _domainMin[1] - 1 && i >= _domainMin[0] && i <= _domainMax[0] - 1 );
        };

        /**
         * Determine whether the cell is on the right boundary
         * @param i Index in x-direction
         * @param j Index in y-direction
         * @param k Index in z-direction
         * @return Returns true if the cell is on the right boundary
         **/
        bool isRightBoundary( const int i, const int j, const int k ) const {
            return ( j == _domainMax[1] && i >= _domainMin[0] && i <= _domainMax[0] - 1 );
        };

        /**
         * Determine whether the cell is on the bottom boundary
         * @param i Index in x-direction
         * @param j Index in y-direction
         * @param k Index in z-direction
         * @return Returns true if the cell is on the bottom boundary
         **/
        bool isBottomBoundary( const int i, const int j, const int k ) const {
            return ( i == _domainMax[0] && j >= _domainMin[1] && j <= _domainMax[1] - 1 );
        };

        /**
         * Determine whether the cell is on the top boundary
         * @param i Index in x-direction
         * @param j Index in y-direction
         * @param k Index in z-direction
         * @return Returns true if the cell is on the top boundary
         **/
        bool isTopBoundary( const int i, const int j, const int k ) const {
            return ( i == _domainMin[0] - 1 && j >= _domainMin[1] && j <= _domainMax[1] - 1 );
        };

    private:
        int _rank;                                                                      /**< Rank of the mesh */
        std::shared_ptr<Cajita::LocalGrid<Cajita::UniformMesh<state_t>>> _local_grid;   /**< Cajita Local Grid */
        std::array<long, 3> _domainMin;                                                 /**< Indices of lower corner of domain */
        std::array<long, 3> _domainMax;                                                 /**< Indices of upper corner of domain */
        const std::array<state_t, 6> _global_bounding_box;                              /**< Array of global bounding box */
};

}

#endif
