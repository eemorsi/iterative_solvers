

/*
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

int hypre_FlexGMRESModifyPCAMG_Func(void *precond_data, int iterations,
                                    double rel_residual_norm);

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

   HYPRE_IJMatrix A;
   HYPRE_ParCSRMatrix parcsr_A;
   HYPRE_IJVector b;
   HYPRE_ParVector par_b;
   HYPRE_IJVector x;
   HYPRE_ParVector par_x;

   int maxit, prnt_info, precond_log, rhs_flag, niter;
   double tol;

   HYPRE_Solver solver, precond;

   /* Initialize MPI */
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

   /* Initialize HYPRE */
   HYPRE_Init();

   double mytime = 0.0;
   double walltime = 0.0;
   maxit = 1000;
   tol = 1e-6;
   prnt_info = 2;
   precond_log = 1;
   niter = 1;
   /* Default problem parameters */
   n = 33;
   rhs_flag = 0;
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
         else if (strcmp(argv[arg_index], "-log") == 0)
         {
            arg_index++;
            prnt_info = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-niter") == 0)
         {
            arg_index++;
            niter = atoi(argv[arg_index++]);
         }
         else if (strcmp(argv[arg_index], "-precond_log") == 0)
         {
            arg_index++;
            precond_log = atoi(argv[arg_index++]);
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
            rhs_flag = 1;
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

      if ((print_usage) && (myid == 0))
      {
         printf("\n");
         printf("Usage: %s [<options>]\n", argv[0]);
         printf("\n");
         printf("  -mtx <matrix.mtx>              : matrix market path\n");
         printf("  -b <vector.mtx>              : RHS matrix market format path\n");
         printf("  -maxit <n>           : Max number of iterations\n");
         printf("  -tol <n>           : tolerance\n");
         printf("  -log <n>           : printing level out of the solver\n");
         printf("  -precond_log <n>           : printing level out of the preconditioner\n");
         printf("  -niter <n>           : solve problem n times\n");
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
         MPI_Finalize();
         return (0);
      }
   }

   int rowindx[num_procs];
   // int local_nnz[num_procs];

   if (myid == 0)
   {
      // SparseMatrixCOO coo_matrix;
      fast_load_from_mtx_file(mtx_filepath, &t_coo_matrix);
      #if 0
      sort_coo_row_padding(&t_coo_matrix, &coo_matrix);
      #else
      sort_coo_row(&t_coo_matrix, &coo_matrix);
      #endif
      if (rhs_flag)
         fast_load_from_array_file(mtx_b_filepath, &rhs);

      size_t local_chunk = coo_matrix.nnz / num_procs;

      j = 0;
      size_t chunk = local_chunk;

      for (i = 0; i < num_procs; i++)
      {
         HYPRE_Int tmp = coo_matrix.rows[chunk];
         while (coo_matrix.rows[chunk] == tmp)
         {
            chunk++;
         }

         rowindx[i] = chunk - 1;

         // printf("Proc %d\t row indx %d \t %d\n", i , chunk, tmp);
         chunk += local_chunk;
      }
      /* remove empty lines errors*/
      rowindx[num_procs - 1] = coo_matrix.nnz - 1;
   }
   MPI_Bcast(rowindx, num_procs, MPI_INT, 0, MPI_COMM_WORLD);
   // MPI_Bcast(local_nnz, num_procs, MPI_INT, 0, MPI_COMM_WORLD);

   MPI_Bcast(&coo_matrix.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&coo_matrix.nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(&coo_matrix.ncolumns, 1, MPI_INT, 0, MPI_COMM_WORLD);

   if (myid != 0)
   {
      if (rhs_flag)
         rhs = (double *)aligned_alloc(align_size, coo_matrix.nrows * sizeof(double));
      coo_matrix.values = (double *)aligned_alloc(align_size, coo_matrix.nnz * sizeof(double));
      coo_matrix.columns = (int *)aligned_alloc(align_size, coo_matrix.nnz * sizeof(int));
      coo_matrix.rows = (int *)aligned_alloc(align_size, coo_matrix.nnz * sizeof(int));
   }
   if (rhs_flag)
      MPI_Bcast(rhs, coo_matrix.nrows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(coo_matrix.values, coo_matrix.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(coo_matrix.columns, coo_matrix.nnz, MPI_INT, 0, MPI_COMM_WORLD);
   MPI_Bcast(coo_matrix.rows, coo_matrix.nnz, MPI_INT, 0, MPI_COMM_WORLD);

   n = coo_matrix.nrows;
   nnz = coo_matrix.nnz;

   // printf("%d\t --> %d\t %d\t %f\n", myid, coo_matrix.rows[rowindx[myid]], coo_matrix.columns[rowindx[myid]], coo_matrix.values[rowindx[myid]]);

   // MPI_Finalize();
   //    return (0);

   // if (myid != 0)
   // {
   //    for ( i = 0; i < coo_matrix.nnz; i++)
   //    {
   //       printf("%d,%d --> %f\n", coo_matrix.rows[i], coo_matrix.columns[i], coo_matrix.values[i]);
   //    }

   // }
   // else
   // {
   //    printf("%d\t %d --> %d\n", myid, 0, rowindx[myid]);
   // }

   // MPI_Finalize();
   // return (0);

   /* Preliminaries: want at least one processor per row */
   if (n * n < num_procs)
      n = sqrt(num_procs) + 1;

   N = coo_matrix.nrows; /* global number of rows */
   h = 1.0 / (n + 1);    /* mesh size*/
   h2 = h * h;

   /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
   // local_size = N / num_procs;
   // extra = N - local_size * num_procs;

   // ilower = local_size * myid;
   // ilower += my_min(myid, extra);

   // iupper = local_size * (myid + 1);
   // iupper += my_min(myid + 1, extra);
   // iupper = iupper - 1;

   ilower[0] = 0;
   if (myid != 0)
      ilower[0] = coo_matrix.rows[rowindx[myid - 1] + 1];

   ilower[1] = 0;

   iupper[0] = coo_matrix.rows[rowindx[myid] - 1];

   iupper[1] = coo_matrix.ncolumns - 1;

   // printf("%d\t %d, %d \t %d\n", myid, ilower[0], iupper[0], rowindx[myid]);

   // printf("The number of cols %d, %d\n", illower[1], iupper[1]  );

   /* How many rows do I have? */
   local_size = iupper[0] - ilower[0] + 1;

   // printf("%d\t%d -> %d  @ %d\n", myid, ilower[0], iupper[0], coo_matrix.rows[ilower[0]]);

   /* Create the matrix. */
   // printf("%d -> %d, %d, %d, %d\n", myid, ilower[0], iupper[0], ilower[1], iupper[1]);
   HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower[0], iupper[0], ilower[0], iupper[0], &A);

   /* Choose a parallel csr format storage */
   HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
   HYPRE_Int omp_flag = 1;
   /*if set, will use a threaded version of HYPRE_IJMatrixSetValues and HYPRE_IJMatrixAddToValues*/
   HYPRE_IJMatrixSetOMPFlag(A, omp_flag);

   /* Initialize before setting coefficients */
   HYPRE_IJMatrixInitialize(A);

   // HYPRE_Int nrows = iupper[0] - ilower[0];

   HYPRE_Int *ncols, *rows;
   ncols = (HYPRE_Int *)aligned_alloc(align_size, local_size * sizeof(HYPRE_Int));
   rows = (HYPRE_Int *)aligned_alloc(align_size, local_size * sizeof(HYPRE_Int));

   HYPRE_Int l_idx = 0;
   if (num_procs > 1)
      l_idx = rowindx[myid - 1] + 1;

#if 1
   for (i = ilower[0]; i <= iupper[0]; i++)
   {
      rows[i - ilower[0]] = i;
      ncols[i - ilower[0]] = 0;
   }
   for (i = l_idx; i < rowindx[myid]; i++)
   {
      ncols[coo_matrix.rows[i] - ilower[0]]++;
   }
   // if (myid == 1)
   //    for (i = 0; i < local_size; i++)
   //    {
   //       printf("%d \t %d\n", rows[i], ncols[i]);
   //    }

   // l_idx = rowindx[myid - 1] + 1;
   HYPRE_IJMatrixSetValues(A, local_size, ncols, rows, &coo_matrix.columns[l_idx], &coo_matrix.values[l_idx]);

   free(ncols);
   free(rows);
#else
   memset(ncols, 0, local_size);
   for (i = l_idx; i < rowindx[myid]; i++)
   {
      ncols[coo_matrix.rows[i] - ilower[0]]++;
   }

   for (i = ilower[0]; i <= iupper[0]; i++)
   {
      l_nnz = ncols[i - ilower[0]];
      // printf("l_nnz @%d\t %d \n", i, l_nnz);
      double values[l_nnz];
      HYPRE_Int cols[l_nnz];
      for (j = 0; j < l_nnz; j++)
      {
         values[j] = coo_matrix.values[l_idx];
         cols[j] = coo_matrix.columns[l_idx];
         l_idx++;
      }
      HYPRE_IJMatrixSetValues(A, 1, &l_nnz, &i, cols, values);
      // l_idx+=l_nnz;
   }

#endif

   /* Assemble after setting the coefficients */
   HYPRE_IJMatrixAssemble(A);

   /* Get the parcsr matrix object to use */
   HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);

   /* Create the rhs and solution */

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower[0], iupper[0], &b);
   HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(b);

   HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower[0], iupper[0], &x);
   HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
   HYPRE_IJVectorInitialize(x);

   /* Set the rhs values and the solution to zero */
   {
      double *x_values, *rhs_values;
      // HYPRE_Int *rows;

      printf("The total number of cols is %d\n", coo_matrix.ncolumns);
      
      rhs_values = (double *)calloc(local_size, sizeof(double));
      x_values = (double *)calloc(local_size, sizeof(double));
      rows = (HYPRE_Int *)calloc(local_size, sizeof(HYPRE_Int));

      for (i = 0; i < local_size; i++)
      {
         // if(isnan(rhs[i]))
         //    printf("NAN %d", i);
         // printf("%f\n", rhs[ilower[0]+1]);
         rhs_values[i] = 1;
         if (rhs_flag)
            rhs_values[i] = rhs[ilower[0] + i];
         x_values[i] = 0.0;
         rows[i] = ilower[0] + i;
      }

      HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
      HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

      free(x_values);
      free(rhs_values);
      free(rows);
   }

   HYPRE_IJVectorAssemble(b);
   /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
   HYPRE_IJVectorGetObject(b, (void **)&par_b);

   HYPRE_IJVectorAssemble(x);
   HYPRE_IJVectorGetObject(x, (void **)&par_x);

   // MPI_Finalize();
   //    return (0);

   /*  Print out the system  - files names will be IJ.out.A.XXXXX
       and IJ.out.b.XXXXX, where XXXXX = processor id */
   if (print_system)
   {
      HYPRE_IJMatrixPrint(A, "IJ.out.A");
      HYPRE_IJVectorPrint(b, "IJ.out.b");
   }

   // MPI_Finalize();
   // return (0);

   /* Choose a solver and solve the system */
   int restart = 30;

   // printf("SOLVER ID %d %lg \n ", solver_id, tol);

   /* AMG */
   if (solver_id == 0)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_BoomerAMGCreate(&solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_BoomerAMGSetPrintLevel(solver, prnt_info); /* print solve info + parameters */
      HYPRE_BoomerAMGSetOldDefault(solver);            /* Falgout coarsening with modified classical interpolation */
      HYPRE_BoomerAMGSetRelaxType(solver, 3);          /* G-S/Jacobi hybrid relaxation */
      HYPRE_BoomerAMGSetRelaxOrder(solver, 0);         /* Uses C/F relaxation */
      HYPRE_BoomerAMGSetNumSweeps(solver, 1);          /* Sweeeps on each level */
      HYPRE_BoomerAMGSetMaxLevels(solver, 20);         /* maximum number of levels */
      HYPRE_BoomerAMGSetMaxIter(solver, maxit);
      HYPRE_BoomerAMGSetTol(solver, tol); /* conv. tolerance */

      /* Now setup and solve! */
      HYPRE_BoomerAMGSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_BoomerAMGSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nBoomerAMG Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_BoomerAMGGetNumIterations(solver, &num_iterations);
      HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_BoomerAMGDestroy(solver);
   }
   /* BiCGStab*/
   else if (solver_id == 6)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, prnt_info); /* prints out the iteration info */
      HYPRE_ParCSRBiCGSTABSetLogging(solver, 1);            /* needed to get run info later */

      HYPRE_ParCSRBiCGSTABSetTol(solver, tol);       /* conv. tolerance */
      HYPRE_ParCSRBiCGSTABSetMaxIter(solver, maxit); /* max iterations */

      /* Now setup and solve! */
      HYPRE_ParCSRBiCGSTABSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRBiCGSTABSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nParCSRBiCGSTABC Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_ParCSRBiCGSTABDestroy(solver);
   }
   /* BICGStab + ILU */

   else if (solver_id == 9)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParCSRBiCGSTABSetPrintLevel(solver, prnt_info); /* prints out the iteration info */
      HYPRE_ParCSRBiCGSTABSetLogging(solver, 1);            /* needed to get run info later */

      HYPRE_ParCSRBiCGSTABSetTol(solver, tol);       /* conv. tolerance */
      HYPRE_ParCSRBiCGSTABSetMaxIter(solver, maxit); /* max iterations */

      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_ILUCreate(&precond);
      HYPRE_ILUSetPrintLevel(precond, precond_log); /* print amg solution info */

      HYPRE_ILUSetTol(precond, 0.0);   /* conv. tolerance zero */
      HYPRE_ILUSetMaxIter(precond, 1); /* do only one iteration! */

      /**  
       * ILUSetType: 
         0 : BJ with ILU(k) (default, with k = 0)
         1 : BJ with ILUT
         10 : GMRES with ILU(k)
         11 : GMRES with ILUT
         20 : NSH with ILU(k)
         21 : NSH with ILUT
         30 : RAS with ILU(k)
         31 : RAS with ILUT
         40 : (nonsymmetric permutation) DDPQ-GMRES with ILU(k)
         41 : (nonsymmetric permutation) DDPQ-GMRES with ILUT
         50 : GMRES with RAP-ILU(0) using MILU(0) for P
       */
      HYPRE_ILUSetType(precond, 10);

      /* Set the PCG preconditioner */
      HYPRE_ParCSRBiCGSTABSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_ILUSolve,
                                     (HYPRE_PtrToParSolverFcn)HYPRE_ILUSetup, precond);

      /* Now setup and solve! */
      HYPRE_ParCSRBiCGSTABSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRBiCGSTABSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nParCSRBiCGSTABC Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_ParCSRBiCGSTABGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_ILUDestroy(precond);
      HYPRE_ParCSRBiCGSTABDestroy(solver);
   }
   /* PCG */
   else if (solver_id == 50)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit);        /* max iterations */
      HYPRE_PCGSetTol(solver, tol);              /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1);            /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, prnt_info); /* prints out the iteration info */
      HYPRE_PCGSetLogging(solver, 1);            /* needed to get run info later */

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nPCG Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver */
      HYPRE_ParCSRPCGDestroy(solver);
   }
   /* PCG with AMG preconditioner */
   else if (solver_id == 1)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParCSRPCGSetMaxIter(solver, maxit);        /* max iterations */
      HYPRE_ParCSRPCGSetTol(solver, tol);              /* conv. tolerance */
      HYPRE_ParCSRPCGSetTwoNorm(solver, 1);            /* use the two norm as the stopping criteria */
      HYPRE_ParCSRPCGSetPrintLevel(solver, prnt_info); /* print solve info */
      HYPRE_ParCSRPCGSetLogging(solver, 1);            /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);

      HYPRE_BoomerAMGSetPrintLevel(precond, precond_log); /* print amg solution info */
      /*
      0 : CLJP-coarsening (a parallel coarsening algorithm using independent sets.
      1 : classical Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
      3 : classical Ruge-Stueben coarsening on each processor, followed by a third pass, which adds coarse points on the boundaries
      6 : Falgout coarsening (uses 1 first, followed by CLJP using the interior coarse points generated by 1 as its first independent set)
      7 : CLJP-coarsening (using a fixed random vector, for debugging purposes only)
      8 : PMIS-coarsening (a parallel coarsening algorithm using independent sets, generating lower complexities than CLJP, might also lead to slower convergence)
      9 : PMIS-coarsening (using a fixed random vector, for debugging purposes only)
      10 : HMIS-coarsening (uses one pass Ruge-Stueben on each processor independently, followed by PMIS using the interior C-points generated as its first independent set)
      11 : one-pass Ruge-Stueben coarsening on each processor, no boundary treatment (not recommended!)
      21 : CGC coarsening by M. Griebel, B. Metsch and A. Schweitzer
      22 : CGC-E coarsening by M. Griebel, B. Metsch and A.Schweitzer
      */
      HYPRE_BoomerAMGSetCoarsenType(precond, 0);
      HYPRE_BoomerAMGSetOldDefault(precond);

      // HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.01);  /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the PCG preconditioner */
      HYPRE_ParCSRPCGSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, precond);
      // HYPRE_ParCSRFlexGMRESSetModifyPC(solver,
      //                                     (HYPRE_PtrToModifyPCFcn)hypre_FlexGMRESModifyPCAMG_Func);

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nPCG + BoomerAMG Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destroy solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_BoomerAMGDestroy(precond);
   }
   /* GMRES + AMG */
   else if (solver_id == 7)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;
      /* Start timing */
      mytime -= MPI_Wtime();

      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

      HYPRE_ParCSRGMRESSetMaxIter(solver, maxit);
      HYPRE_ParCSRGMRESSetKDim(solver, restart);
      HYPRE_ParCSRGMRESSetTol(solver, tol);

      HYPRE_ParCSRGMRESSetPrintLevel(solver, prnt_info);
      HYPRE_ParCSRGMRESSetLogging(solver, 1);
      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, precond_log); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetOldDefault(precond);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0);   /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      HYPRE_ParCSRGMRESSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                                  (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, precond);

      /* Call the setup */
      // HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);
      HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);

      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Setup time = %f seconds\n\n", walltime);
      }

      /* Start timing again */
      mytime -= MPI_Wtime();

      /* Call the solve */
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);

      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Solve time = %f seconds\n\n", walltime);
      }

      /* Get some info */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
      /* Clean up */
      HYPRE_BoomerAMGDestroy(precond);
      // HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_ParCSRGMRESDestroy(solver);
   }

   /* GMRES + ILU */
   else if (solver_id == 11)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;
      /* Start timing */
      mytime -= MPI_Wtime();

      HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &solver);

      HYPRE_ParCSRGMRESSetMaxIter(solver, maxit);
      HYPRE_ParCSRGMRESSetKDim(solver, restart);
      HYPRE_ParCSRGMRESSetTol(solver, tol);

      HYPRE_ParCSRGMRESSetPrintLevel(solver, prnt_info);
      HYPRE_ParCSRGMRESSetLogging(solver, 1);

      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_ILUCreate(&precond);
      HYPRE_ILUSetPrintLevel(precond, precond_log); /* print amg solution info */

      HYPRE_ILUSetTol(precond, 0.0);   /* conv. tolerance zero */
      HYPRE_ILUSetMaxIter(precond, 1); /* do only one iteration! */

      /**  
       * ILUSetType: 
         0 : BJ with ILU(k) (default, with k = 0)
         1 : BJ with ILUT
         10 : GMRES with ILU(k)
         11 : GMRES with ILUT
         20 : NSH with ILU(k)
         21 : NSH with ILUT
         30 : RAS with ILU(k)
         31 : RAS with ILUT
         40 : (nonsymmetric permutation) DDPQ-GMRES with ILU(k)
         41 : (nonsymmetric permutation) DDPQ-GMRES with ILUT
         50 : GMRES with RAP-ILU(0) using MILU(0) for P
       */
      HYPRE_ILUSetType(precond, 11);

      HYPRE_ParCSRGMRESSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_ILUSolve,
                                  (HYPRE_PtrToParSolverFcn)HYPRE_ILUSetup, precond);

      /* Call the setup */
      // HYPRE_ParCSRPCGSetup(solver, par_A, par_b, par_x);
      HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);

      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Setup time = %f seconds\n\n", walltime);
      }

      /* Start timing again */
      mytime -= MPI_Wtime();

      /* Call the solve */
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);

      /* Finalize current timing */
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Solve time = %f seconds\n\n", walltime);
      }

      /* Get some info */
      HYPRE_GMRESGetNumIterations(solver, &num_iterations);
      HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }
      /* Clean up */
      // HYPRE_ILUDestroy(precond);
      HYPRE_ParCSRGMRESDestroy(solver);
   }

   /* PCG with Parasails Preconditioner */
   else if (solver_id == 8)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;

      int sai_max_levels = 1;
      double sai_threshold = 0.1;
      double sai_filter = 0.05;
      int sai_sym = 1;

      /* Create solver */
      HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_PCGSetMaxIter(solver, maxit);        /* max iterations */
      HYPRE_PCGSetTol(solver, tol);              /* conv. tolerance */
      HYPRE_PCGSetTwoNorm(solver, 1);            /* use the two norm as the stopping criteria */
      HYPRE_PCGSetPrintLevel(solver, prnt_info); /* print solve info */
      HYPRE_PCGSetLogging(solver, 1);            /* needed to get run info later */

      /* Now set up the ParaSails preconditioner and specify any parameters */
      HYPRE_ParaSailsCreate(MPI_COMM_WORLD, &precond);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParaSailsSetParams(precond, sai_threshold, sai_max_levels);
      HYPRE_ParaSailsSetFilter(precond, sai_filter);
      HYPRE_ParaSailsSetSym(precond, sai_sym);
      HYPRE_ParaSailsSetLogging(precond, 3);

      /* Set the PCG preconditioner */
      HYPRE_PCGSetPrecond(solver, (HYPRE_PtrToSolverFcn)HYPRE_ParaSailsSolve,
                          (HYPRE_PtrToSolverFcn)HYPRE_ParaSailsSetup, precond);

      /* Now setup and solve! */
      HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nPCG +ParaSail Solve time = %f seconds\n\n", walltime);
      }
      /* Run info - needed logging turned on */
      HYPRE_PCGGetNumIterations(solver, &num_iterations);
      HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRPCGDestroy(solver);
      HYPRE_ParaSailsDestroy(precond);
   }
   /* Flexible GMRES with  AMG Preconditioner */
   else if (solver_id == 61)
   {
      HYPRE_Int num_iterations;
      double final_res_norm;
      int modify = 1;

      /* Create solver */
      HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_ParCSRFlexGMRESSetKDim(solver, restart);
      HYPRE_ParCSRFlexGMRESSetMaxIter(solver, maxit);        /* max iterations */
      HYPRE_ParCSRFlexGMRESSetTol(solver, tol);              /* conv. tolerance */
      HYPRE_ParCSRFlexGMRESSetPrintLevel(solver, prnt_info); /* print solve info */
      HYPRE_ParCSRFlexGMRESSetLogging(solver, 1);            /* needed to get run info later */

      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, precond_log); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 6);
      HYPRE_BoomerAMGSetOldDefault(precond);
      HYPRE_BoomerAMGSetRelaxType(precond, 6); /* Sym G.S./Jacobi hybrid */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0);   /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the FlexGMRES preconditioner */
      HYPRE_ParCSRFlexGMRESSetPrecond(solver, (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSolve,
                                      (HYPRE_PtrToParSolverFcn)HYPRE_BoomerAMGSetup, precond);

      if (modify)
         /* this is an optional call  - if you don't call it, hypre_FlexGMRESModifyPCDefault
         is used - which does nothing.  Otherwise, you can define your own, similar to
         the one used here */
         HYPRE_ParCSRFlexGMRESSetModifyPC(solver,
                                          (HYPRE_PtrToModifyPCFcn)hypre_FlexGMRESModifyPCAMG_Func);

      /* Now setup and solve! */
      HYPRE_ParCSRFlexGMRESSetup(solver, parcsr_A, par_b, par_x);

      // MPI_Finalize();
      // return (0);

      /* Start timing again */
      mytime -= MPI_Wtime();
      for (i = 0; i < niter; i++)
         HYPRE_ParCSRFlexGMRESSolve(solver, parcsr_A, par_b, par_x);
      mytime += MPI_Wtime();
      MPI_Allreduce(&mytime, &walltime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (myid == 0)
      {
         printf("\nGMRES Solve time = %f seconds\n\n", walltime);
      }

      /* Run info - needed logging turned on */
      HYPRE_ParCSRFlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
      if (myid == 0)
      {
         printf("\n");
         printf("Iterations = %lld\n", num_iterations);
         printf("Final Relative Residual Norm = %e\n", final_res_norm);
         printf("\n");
      }

      /* Destory solver and preconditioner */
      HYPRE_ParCSRFlexGMRESDestroy(solver);
#if 0
   HYPRE_BoomerAMGDestroy(precond);
#endif
   }
   else
   {
      if (myid == 0)
         printf("Invalid solver id specified.\n");
   }

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

/*--------------------------------------------------------------------------
   hypre_FlexGMRESModifyPCAMG_Func -

    This is an example (not recommended)
   of how we can modify things about AMG that
   affect the solve phase based on how FlexGMRES is doing...For
   another preconditioner it may make sense to modify the tolerance..

 *--------------------------------------------------------------------------*/

int hypre_FlexGMRESModifyPCAMG_Func(void *precond_data, int iterations,
                                    double rel_residual_norm)
{

   if (rel_residual_norm > .1)
   {
      HYPRE_BoomerAMGSetNumSweeps((HYPRE_Solver)precond_data, 10);
   }
   else
   {
      HYPRE_BoomerAMGSetNumSweeps((HYPRE_Solver)precond_data, 1);
   }

   return 0;
}
