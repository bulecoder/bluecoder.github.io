# Python基础
## 魔法方法（[学习视频地址](https://www.bilibili.com/video/BV1b84y1e7hG?spm_id_from=333.788.videopod.sections&vd_source=d3285a2ba86bc368a3901aac90d388ea)）
* __new__: 从class建立一个object的过程;
    例如想要做一个singleton class（单例类），在建立一个class的object之前，判断一下有没有其他object已经被建立；还有需要用到metaclass有关的内容也会用到new；new有返回值，返回这个object；
* __init__: 有object之后给object初始化的过程;
    初始化使用 init；init没有返回值；
* __del__: 可以理解为析构函数，但不是析构函数;
    当对象被释放的时候需要干什么; `__del__`和python里面的关键字del没有关系，使用关键字del对象并不一定会触发`__del__`，del对象只是让这个对象少一个引用；只有当对象完全被释放的时候才会触发`__del__`
* __repr__: 返回object的字符串表示;
    返回更详细的信息，给开发者用的；需要显示调用(repr(obj))；`__repr__`找不到不会去找`__str__`;
* __str__:返回object的字符串表示;
    返回人类更容易理解的string，注重可读性；打印对象/字符串时被调用;`__str__`找不到会自动去找`__repr__`;
* __format__: 使用某种格式打印object；
    当使用 `f"{obj:格式}"` 或者
    ```
    "{}".format(obj)
    "{:格式}".format(obj)
    format(obj, "格式")
    ```
    的时候会被调用，如果找不到会去找`__str__`
* __bytes__：客制化object的bytes表示;
    bytes(obj)显示调用
### rich comparison
* __eq__：使用"=="的时候被调用，如果不写__eq__方法，使用"=="的时候，其实是使用"is"在判断;
* __ne__：使用"!="的时候被调用，如果不写__ne__方法，使用"!="的时候，其实是使用"is"取反的逻辑;
* __gt__：使用">"的时候被调用，"<"逻辑直接取反（__ge__同理）;
* __lt__：使用"<"的时候被调用，对于这类rich comparison的函数（要比较x和y的大小，如果y是x的衍生类，则优先使用y的rich comparison，否则优先使用x的rich comparison）（__le__同理，但是python不会推理，不会认为小于等于就是小于或者等于，即 x <= y  不等价于 x < y or x == y）;
* __hash__：python会对自定义类默认__eq__和__hash__，但是如果对自定义类定义了自己的__eq__函数，默认的__hash__函数就会被删掉，__hash__的基本定义在于如果两个东西相等，hash值必须相等。__hash__的要求是必须返回一个整数，对于两个相等的对象，必须要有同样的hash值
* __bool__：当自定义对象放在条件判断语句中是，默认都是true，只有当自定义了__bool__方法，自定义对象放在条件判断语句中时会被调用;
* __getattr__：读取一个对象不存在的属性的时候才会被调用;
* __getattribute__：只要尝试读取对象的属性，都会被调用，该方法非常容易产生无限递归。default behavior是super().getattribute（或者是object().setattr）;
* __setattr__：尝试去写一个属性的时候，就会被调用;同样是，使用super().__setattr__来完成默认行为（或者使用object.__setattr__来完成）
* __delattr__：尝试删除一个object的属性的时候被调用;
* __dir__：调用dir(object)的时候被调用，必须返回一个sequence;
* __slots__：不是一个魔术方法，__slots__ = ['a', 'b']表示这个class的object里面可以有哪些自定义的attribute，这里表示只能有属性a和b（白名单机制）;
### emulation


