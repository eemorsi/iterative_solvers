

/*
   This implementation is based on ex5big from HYPRE examples 

   Interface:    Linear-Algebraic (IJ)

   Compile with: make

   Sample run:   mpirun -ve 0-1 -np 4 ./solver -print_system -solver 0 -mtx ~/A_1024/A_1024.mtx -b ~/A_1024/A_1024_b.mtx

   Description:  
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "sparse_matrix.h"


#define my_min(a, b) (((a) < (b)) ? (a) : (b))

int main(int argc, char *argv[])
{
   HYPRE_Int i, j;
   int myid, num_procs;
   size_t N, n;
   size_t nnz;

   char *mtx_filepath;
   char *mtx_b_filepath;

   SparseMatrixCOO coo_matrix, t_coo_matrix;
   double *rhs;

   HYPRE_Int ilower[2], iupper[2];
   HYPRE_Int local_size;

   int solver_id;
   int print_system;

   double h, h2;

   HYPRE_IJMatrix ij_A;
   HYPRE_Matrix A;
   HYPRE_IJVector ij_b;
   HYPRE_Vector b;
   HYPRE_IJVector ij_x;
   HYPRE_Vector x;

   int maxit;
   double tol;

   HYPRE_Solver solver, precond;

   /* Initialize HYPRE */
   MPI_Init(&argc, &argv);
   HYPRE_Init();

   double mytime = 0.0;
   double walltime = 0.0;
   maxit = 1000;
   tol = 1e-6;

   /* Default problem parameters */
   n = 33;

   solver_id = 0;
   print_system = 0;

   /* Parse command line */
   {
      int arg_index = 0;
      int print_usage = 0;

      while (arg_index < argc)
      {
         if (strcmp(argv[arg_index], "-maxit") == 0)
         {
            arg_index++;
            maxit = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-tol") == 0)
         {
            arg_index++;
            tol = atof(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-solver") == 0)
         {
            arg_index++;
            solver_id = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-mtx") == 0)
         {
            arg_index++;
            mtx_filepath = argv[arg_index++];
         }
         else if (strcmp(argv[arg_index], "-b") == 0)
         {
            arg_index++;
            mtx_b_filepath = argv[arg_index++];
         }
         else if (strcmp(argv[arg_index], "-print_system") == 0)
         {
            arg_index++;
            print_system = 1;
         }
         else if (strcmp(argv[arg_index], "-help") == 0)
         {
            print_usage = 1;
            break;
         }
         else
         {
            arg_index++;
         }
      }

      if ((print_usage))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -mtx <matrix.mtx>              : matrix market path\n");
         printf("  -b <vector.mtx>              : RHS matrix market format path\n");
         printf("  -maxit <n>           : Max number of iterations\n");
         printf("  -tol <n>           : tolerance\n");
         printf("  -solver <ID>        : solver ID\n");
         printf("                        0  - AMG (default) \n");
         printf("                        1  - AMG-PCG\n");
         printf("                        6  - BiCGStab\n");
         printf("                        9  - ILU + BiCGStab\n");
         printf("                        7  - AMG-GMRES\n");
         printf("                        8  - ParaSails-PCG\n");
         printf("                        11  - ILU-GMRES\n");
         printf("                        50 - PCG\n");
         printf("                        61 - AMG-FlexGMRES\n");
         printf("  -print_system       : print the matrix and rhs\n");
         printf("\n");
      }

      if (print_usage)
      {
         return (0);
      }
   }


   if (myid == 0)
   {
      // SparseMatrixCOO coo_matrix;
      fast_load_from_mtx_file(mtx_filepath, &t_coo_matrix);
      sort_coo_row(&t_coo_matrix, &coo_matrix);
      fast_load_from_array_file(mtx_b_filepath, &rhs);

   }
   
   n = coo_matrix.nrows;
   nnz = coo_matrix.nnz;

   ilower[0] = 0;

   ilower[1] = coo_matrix.nrows;

   iupper[0] = 0;

   iupper[1] = coo_matrix.ncolumns - 1;

   // printf("%d\t %d, %d \t %d\n", myid, ilower[0], iupper[0], rowindx[myid]);

   // printf("The number of cols %d, %d\n", illower[1], iupper[1]  );

   /* How many rows do I have? */
   local_size = iupper[0] - ilower[0] + 1;

   // printf("%d\t%d -> %d  @ %d\n", myid, ilower[0], iupper[0], coo_matrix.rows[ilower[0]]);

   /* Create the matrix. */
   // printf("%d -> %d, %d, %d, %d\n", myid, ilower[0], iupper[0], ilower[1], iupper[1]);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower[0], iupper[0], ilower[0], iupper[0], &ij_A);

   /* Choose a parallel csr format storage */
   HYPRE_IJMatrixSetObjectType(ij_A, HYPRE_PARCSR);
   // HYPRE_Int omp_flag = 1;
   // HYPRE_IJMatrixSetOMPFlag(A, omp_flag);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(ij_A);

   // HYPRE_Int nrows = iupper[0] - ilower[0];

   HYPRE_Int *ncols, *rows;
   ncols = (HYPRE_Int *)aligned_alloc(align_size, local_size * sizeof(HYPRE_Int));
   rows = (HYPRE_Int *)aligned_alloc(align_size, local_size * sizeof(HYPRE_Int));


   for (i = ilower[0]; i <= iupper[0]; i++)
   {
      rows[i] = i;
      ncols[i] = 0;
   }
   for (i = 0; i < coo_matrix.nnz; i++)
   {
      ncols[coo_matrix.rows[i]]++;
   }
   // if (myid == 1)
   //    for (i = 0; i < local_size; i++)
   //    {
   //       printf("%d \t %d\n", rows[i], ncols[i]);
   //    }

   // l_idx = rowindx[myid - 1] + 1;
   HYPRE_IJMatrixSetValues(ij_A, local_size, ncols, rows, coo_matrix.columns, coo_matrix.values);

   free(ncols);
   free(rows);

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(ij_A);

   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(ij_A, (void **)&A);

   /* Create the rhs and solution */

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower[0], iupper[0], &ij_b);
   HYPRE_IJVectorSetObjectType(ij_b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower[0], iupper[0], &ij_x);
   HYPRE_IJVectorSetObjectType(ij_x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(ij_x);

   /* Set the rhs values and the solution to zero */
   {
      double *x_values, *rhs_values;
      // HYPRE_Int *rows;

      // printf("The total number of cols is %d\n", t_ncol);
      rhs_values = (double *)calloc(local_size, sizeof(double));
      x_values = (double *)calloc(local_size, sizeof(double));
      rows = (HYPRE_Int *)calloc(local_size, sizeof(HYPRE_Int));

      for (i = 0; i < local_size; i++)
      {
         // if(isnan(rhs[i]))
         //    printf("NAN %d", i);
         // printf("%f\n", rhs[ilower[0]+1]);
         rhs_values[i] = rhs[ilower[0] + i];
         x_values[i] = 0.0;
         rows[i] = ilower[0] + i;
      }

      HYPRE_IJVectorSetValues(ij_b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues(ij_x, local_size, rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);
   }

   HYPRE_IJVectorAssemble(ij_b);
   /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
   HYPRE_IJVectorGetObject(ij_b, (void **)&b);

   HYPRE_IJVectorAssemble(ij_x);
   HYPRE_IJVectorGetObject(ij_x, (void **)&x);


   /*  Print out the system  - files names will be IJ.out.A.XXXXX
       and IJ.out.b.XXXXX, where XXXXX = processor id */
   if (print_system)
   {
      HYPRE_IJMatrixPrint(A, "IJ.out.A");
      HYPRE_IJVectorPrint(b, "IJ.out.b");
   }

   // MPI_Finalize();
   // return (0);


   /* Clean up */
   HYPRE_IJMatrixDestroy(A);
   HYPRE_IJVectorDestroy(b);
   HYPRE_IJVectorDestroy(x);

   /* Finalize HYPRE */
   HYPRE_Finalize();

   /* Finalize MPI*/
   MPI_Finalize();

   return (0);
}
