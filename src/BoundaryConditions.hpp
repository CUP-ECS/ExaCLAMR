/**
 * @file
 * @author Patrick Bridges <patrickb@unm.edu>
 * @author Jered Dominguez-Trujillo <jereddt@unm.edu>
 * 
 * @section DESCRIPTION
 * Boundary Conditions for ExaCLAMR Shallow Water Solver
 */

#ifndef EXACLAMR_BOUNDARYCONDITIONS_HPP
#define EXACLAMR_BOUNDARYCONDITIONS_HPP

#ifndef DEBUG
#define DEBUG 0
#endif

// Include Statements
#include <Mesh.hpp>

#include <Kokkos_Core.hpp>

namespace ExaCLAMR {
    /**
 * @struct BoundaryType
 * @brief Struct which contains enums of boundary type options for each boundary
 */
    struct BoundaryType {
        enum Values {
            NONE       = 0,
            REFLECTIVE = 1
        };
    };

    /**
 * @struct BoundaryCondition
 * @brief Struct that applies the specified boundary conditions with a Kokkos Inline Function
 */
    struct BoundaryCondition {
        template <class ProblemManagerType, class ArrayType, class MeshType>
        KOKKOS_INLINE_FUNCTION void operator()( const ProblemManagerType &pm, const int i, const int j, const int k, ArrayType &h_current, ArrayType &u_current, MeshType &mesh ) const {
            // Left Boundary
            if ( mesh.isBottomBoundary( i, j, k ) ) {
                // DEBUG: Print Rank and Bottom Boundary Indices
                // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tBottom Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
                if ( boundary_type[1] == BoundaryType::REFLECTIVE ) {
                    // No Flux Boundary Condition
                    h_current( i, j, k, 0 ) = h_current( i, j + 1, k, 0 );
                    u_current( i, j, k, 0 ) = u_current( i, j + 1, k, 0 );
                    u_current( i, j, k, 1 ) = -u_current( i, j + 1, k, 1 );
                    // Second Boundary Node set to 0
                    h_current( i, j - 1, k, 0 ) = 0;
                    u_current( i, j - 1, k, 0 ) = 0;
                    u_current( i, j - 1, k, 1 ) = 0;
                }
            }

            // Top Boundary
            if ( mesh.isTopBoundary( i, j, k ) ) {
                // DEBUG: Print Rank and Top Boundary Indices
                // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tTop Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
                if ( boundary_type[4] == BoundaryType::REFLECTIVE ) {
                    // No Flux Boundary Condition
                    h_current( i, j, k, 0 ) = h_current( i, j - 1, k, 0 );
                    u_current( i, j, k, 0 ) = u_current( i, j - 1, k, 0 );
                    u_current( i, j, k, 1 ) = -u_current( i, j - 1, k, 1 );
                    // Second Boundary Node set to 0
                    h_current( i, j + 1, k, 0 ) = 0;
                    u_current( i, j + 1, k, 0 ) = 0;
                    u_current( i, j + 1, k, 1 ) = 0;
                }
            }

            // Right Boundary
            if ( mesh.isRightBoundary( i, j, k ) ) {
                // DEBUG: Print Rank and Right Boundary Indices
                // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tRight Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
                if ( boundary_type[3] == BoundaryType::REFLECTIVE ) {
                    // No Flux Boundary Condition
                    h_current( i, j, k, 0 ) = h_current( i - 1, j, k, 0 );
                    u_current( i, j, k, 0 ) = -u_current( i - 1, j, k, 0 );
                    u_current( i, j, k, 1 ) = u_current( i - 1, j, k, 1 );
                    // Second Boundary Node set to 0
                    h_current( i + 1, j, k, 0 ) = 0;
                    u_current( i + 1, j, k, 0 ) = 0;
                    u_current( i + 1, j, k, 1 ) = 0;
                }
            }

            // Left Boundary
            if ( mesh.isLeftBoundary( i, j, k ) ) {
                // DEBUG: Print Rank and Left Boundary Indices
                // if ( DEBUG ) std::cout << "Rank: " << pm.mesh()->rank() << "\tLeft Boundary:\ti: " << i << "\tj: " << j << "\tk: " << k << "\n";
                if ( boundary_type[0] == BoundaryType::REFLECTIVE ) {
                    // No Flux Boundary Condition
                    h_current( i, j, k, 0 ) = h_current( i + 1, j, k, 0 );
                    u_current( i, j, k, 0 ) = -u_current( i + 1, j, k, 0 );
                    u_current( i, j, k, 1 ) = u_current( i + 1, j, k, 1 );
                    // Second Boundary Node set to 0
                    h_current( i - 1, j, k, 0 ) = 0;
                    u_current( i - 1, j, k, 0 ) = 0;
                    u_current( i - 1, j, k, 1 ) = 0;
                }
            }
        }

        Kokkos::Array<int, 6> boundary_type; /**< Boundary condition type on all 6 walls ( 3-D ) */
    };

} // namespace ExaCLAMR

#endif
