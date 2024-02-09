NDK_ROOT ?= $(HOME)/Library/Android/sdk/ndk/26.1.10909125/
ARCH ?= aarch64
# note: this is 386 for x86 AVD -- retest amd64 on x86_64 AVD + update instructions
GOARCH ?= arm64
TARGET_API ?= android28
# onnx version basd on what onnxruntime_go targets
ONNX_VERSION ?= 1.16.1
SO_ARCH ?= arm64-v8a

CGO_ENABLED := 1
CC := $(shell realpath $(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android30-clang)
CGO_CFLAGS := -I$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include \
			                -I$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include/aarch64-linux-android
CGO_LDFLAGS := -L$(NDK_ROOT)/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/lib


module:
	GOOS=android GOARCH=$(GOARCH) CGO_ENABLED=1 \
		CC=$(CC) \
		CGO_CFLAGS="$(CGO_CFLAGS)" \
		CGO_LDFLAGS="$(CGO_LDFLAGS)" \
		go build -v \
		-tags no_cgo \
		-o $@ \
		./cmd/module

onnxruntime-android-$(ONNX_VERSION).aar:
	# see https://onnxruntime.ai/docs/install/#install-on-android
	wget https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$(ONNX_VERSION)/onnxruntime-android-$(ONNX_VERSION).aar

third_party/onnx-android-$(SO_ARCH).so: onnxruntime-android-$(ONNX_VERSION).aar
	unzip -o $< 'jni/*/*.so'
	cp jni/$(SO_ARCH)/libonnxruntime.so $@

bundle-droid-$(SO_ARCH).tar.gz: module third_party/onnx-android-$(SO_ARCH).so
	tar -czf $@ $^
