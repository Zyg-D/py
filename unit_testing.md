Bus testuojami tik 2 metodai (`accelerate` ir `brake`) is sios klases:
```py
class Car:

    def __init__(self, speed=0):
        self.speed = speed
        self.odometer = 0
        self.time = 0

    def accelerate(self):
        self.speed += 5

    def brake(self):
        if self.speed < 5:
            self.speed = 0
        else:
            self.speed -= 5
```

Kiekvienam metodui istestuoti atskirame faile kuriamos atskiros klases. Failas buna daug ilgesnis, nes kiekvienas testuotinas metodas turi po kelis test cases, kurie nuseda i test failo metodus. 
```py
import unittest

from Car import Car


class TestCar(unittest.TestCase):
      def setUp(self):
          self.car = Car()


class TestInit(TestCar):
      def test_initial_speed(self):
          self.assertEqual(self.car.speed, 0)

      def test_initial_odometer(self):
          self.assertEqual(self.car.odometer, 0)

      def test_initial_time(self):
          self.assertEqual(self.car.time, 0)


class TestAccelerate(TestCar):
      def test_accelerate_from_zero(self):
          self.car.accelerate()
          self.assertEqual(self.car.speed, 5)

      def test_multiple_accelerates(self):
          for _ in range(3):
            self.car.accelerate()
          self.assertEqual(self.car.speed, 15)


class TestBrake(TestCar):
       def test_brake_once(self):
           self.car.accelerate()
           self.car.brake()
           self.assertEqual(self.car.speed, 0)

       def test_multiple_brakes(self):
            for _ in range(5):
                self.car.accelerate()
            for _ in range(3):
                self.car.brake()
            self.assertEqual(self.car.speed, 10)

       def test_should_not_allow_negative_speed(self):
           self.car.brake()
           self.assertEqual(self.car.speed, 0)

       def test_multiple_brakes_at_zero(self):
           for _ in range(3):
               self.car.brake()
           self.assertEqual(self.car.speed, 0)
```

