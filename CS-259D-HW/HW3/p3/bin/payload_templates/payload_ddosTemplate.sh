#! /bin/sh

intervalInSecs=
durationInSecs=
ddosTarget=""

function daemonhelper {
    ping -i $intervalInSecs -w $durationInSecs $ddosTarget > /dev/null 2>&1
}

case "$1" in	
  daemon)
      daemonhelper
      ;;

  *)
      $0 daemon
      ;;
esac

exit 0