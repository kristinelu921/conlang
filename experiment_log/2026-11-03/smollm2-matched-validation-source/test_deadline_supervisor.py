"""Verify deadline handling using mocked process handles without starting any job."""

import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from run_remote_phase import supervise_process


class DeadlineSupervisorTests(unittest.TestCase):
    def test_completed_process_is_never_signaled(self) -> None:
        process = Mock(pid=123, spec=subprocess.Popen)
        process.wait.return_value = 0
        with patch("run_remote_phase.os.killpg") as signal_group:
            self.assertEqual(supervise_process(process, 100), (0, False))
            signal_group.assert_not_called()
        process.wait.assert_called_once_with(timeout=90)

    def test_timeout_signals_only_the_owned_group(self) -> None:
        process = Mock(pid=123, spec=subprocess.Popen)
        process.wait.side_effect = [subprocess.TimeoutExpired("phase", 90), -15]
        process.poll.return_value = None
        with patch("run_remote_phase.os.killpg") as signal_group:
            self.assertEqual(supervise_process(process, 100), (-15, True))
            signal_group.assert_called_once_with(123, signal.SIGTERM)

    def test_term_timeout_rechecks_before_kill(self) -> None:
        process = Mock(pid=123, spec=subprocess.Popen)
        process.wait.side_effect = [subprocess.TimeoutExpired("phase", 90),
                                    subprocess.TimeoutExpired("phase", 5), -9]
        process.poll.return_value = None
        with patch("run_remote_phase.os.killpg") as signal_group:
            self.assertEqual(supervise_process(process, 100), (-9, True))
            self.assertEqual([call.args for call in signal_group.call_args_list],
                             [(123, signal.SIGTERM), (123, signal.SIGKILL)])
        self.assertEqual(process.poll.call_count, 2)

    def test_exit_race_does_not_signal_a_finished_process(self) -> None:
        process = Mock(pid=123, spec=subprocess.Popen)
        process.wait.side_effect = [subprocess.TimeoutExpired("phase", 90), 0]
        process.poll.return_value = 0
        with patch("run_remote_phase.os.killpg") as signal_group:
            self.assertEqual(supervise_process(process, 100), (0, True))
            signal_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
