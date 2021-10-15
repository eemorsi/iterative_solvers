
#ifdef __POWER
#include <errno.h>
#include <signal.h>
#include <string.h>
#include <sys/prctl.h>
#include <unistd.h>

#define READ 0
#define WRITE 1

int comp_elem;

FILE *c_popen(char *command, char type, pid_t *pid) {
  pid_t child_pid;
  int fd[2];
  pipe(fd);

  if ((child_pid = fork()) == -1) {
    perror("fork");
    exit(1);
  }

  /* child process */
  if (child_pid == 0) {
    if (type == 'r') {
      close(fd[0]);   // Close the READ end of the pipe since the child's fd is
                      // write-only
      dup2(fd[1], 1); // Redirect stdout to pipe
    } else {
      close(fd[1]);   // Close the WRITE end of the pipe since the child's fd is
                      // read-only
      dup2(fd[0], 0); // Redirect stdin to pipe
    }

    // int r = prctl(PR_SET_PDEATHSIG, SIGKILL);
    // if (r == -1) { perror(0); exit(1); }
    // printf("child pid %d\n", child_pid);
    setpgid(child_pid,
            child_pid); // Needed so negative PIDs can kill children of /bin/sh
    execl("/bin/sh", "sh", "-c", command, (char *)0);
    // system(command);
    // execv(command, (char *)0);

    exit(0);
  } else {

    printf("child pid %d\n", child_pid);

    if (type == 'r') {
      close(fd[1]); // Close the WRITE end of the pipe since parent's fd is
                    // read-only
    } else {
      close(fd[0]); // Close the READ end of the pipe since parent's fd is
                    // write-only
    }
  }

  *pid = child_pid;

  if (type == 'r') {
    return fdopen(fd[0], "r");
  }

  return fdopen(fd[1], "w");
}

int c_pclose(FILE *fp, pid_t pid) {
  int stat;
  char cmd[1000];

  sprintf(cmd, "kill -9 %d", pid);
  // execl("/bin/sh", "sh", "-c", cmd, (char *)0);
  system(cmd);
  // prctl(PR_SET_PDEATHSIG, SIGTERM);
  fclose(fp);
  // while (waitpid(pid, &stat, 0) == -1) {
  //   if (errno != EINTR) {
  //     stat = -1;
  //     break;
  //   }
  // }

  return stat;
}

void get_bash_cmd(char **cmd, char *pow_filepath, int comp_id) {
   pid_t ppid = getpid();
#ifdef __ve__
  // sprintf(bash_cmd,
  //         "J=`ps -p %d|grep %d|wc -l`;while [ $J -ne 0 ]; do "
  //         "/opt/nec/ve/bin/vecmd -N %d info | egrep ^Current -A2 | grep -v
  //         " "Current | awk '{sum=sum + 12 * $5 / 1000}END{print sum}' >>
  //         %s; " "J=`ps -p %d|grep %d|wc -l`;" "sleep 1 ; done &", ppid,
  //         ppid, comp_id, pow_filepath, ppid, ppid);
  sprintf(
      *cmd,
      "J=`ps -p %d|grep %d|wc -l`;while [ $J -ne 0 ]; do "
      "awk -F, '{ getline v1 < \"/sys/class/ve/ve%d/sensor_8\"; v1=v1/1000000 "
      "; getline v2 < \"/sys/class/ve/ve%d/sensor_9\"; v2=v2/1000000; getline "
      "v3 < \"/sys/class/ve/ve%d/sensor_13\";$0=$0*v1+v2*v3; $0=$0/1000; "
      "$0+=5.0; "
      "print $0}' /sys/class/ve/ve%d/sensor_12 >> %s; "
      "J=`ps -p %d|grep %d|wc -l`;"
      "sleep 1 ; done &",
      ppid, ppid, comp_id, comp_id, comp_id, comp_id, pow_filepath, ppid, ppid);
#else
  // --loop-ms=1000
  sprintf(*cmd,
          "J=`ps -p %d|grep %d|wc -l`;while [ $J -ne 0 ]; do "
          "nvidia-smi -i %d --format=csv --query-gpu=power.draw|cut -d\" \" "
          "-f 1|sort -rn|sed 2d >> %s; "
          "J=`ps -p %d|grep %d|wc -l`;"
          "sleep 1 ; done &",
          ppid, ppid, comp_id, pow_filepath, ppid, ppid);
#endif
  printf("%s\n", cmd);
}
#endif